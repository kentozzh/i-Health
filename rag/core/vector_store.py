# 核心流程: 文档向量生成(BGE-M3) -> 向量入库(Milvus) -> 混合检索(稠密 + 稀疏向量) -> 结果去重 -> 重排序 -> 返回精准文档.

import warnings

import pymilvus

warnings.filterwarnings("ignore")

from base import logger, Config

from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import hashlib
import torch.cuda
from rag.core.document_preprocessor import *
import os, sys

rag_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class VectorStore():
    def __init__(
            self,
            database_name=Config().MILVUS_DATABASE_NAME,
            collection_name=Config().MILVUS_COLLECTION_NAME,
            host=Config().MILVUS_HOST,
            port=Config().MILVUS_PORT
    ):
        self.database = database_name
        self.collection = collection_name
        self.host = host
        self.port = port
        self.logger = logger

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备: {self.device}')

        reranker_path = os.path.join(rag_path, 'models/bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        bgem3_path = os.path.join(rag_path, 'models/bge-m3')
        try:
            self.embedding_function = BGEM3EmbeddingFunction(
                model_name_or_path=bgem3_path,
                device=self.device,
                use_fp16=(self.device == 'cuda')
            )
        except Exception as e:
            logger.error(f'初始化向量生成器失败: {e}')

        self.dense_dim = self.embedding_function.dim['dense']
        # print(f'dense向量维度: {self.dense_dim}')

        try:
            self.client = MilvusClient(host=self.host, port=self.port, db_name=self.database)
            logger.info(f'Milvus客户端已初始化: {self.client}')
        except pymilvus.MilvusException as e:
            logger.error(f'初始化Milvus客户端失败: {e}')
            raise

        self.__create_or_load_collection()



    def __create_or_load_collection(self):
        if not self.client.has_collection(self.collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="q_type", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name='dense_vector',
                index_name='dense_index',
                index_type='IVF_FLAT',
                params={'nlist': 128},
                metric_type='IP'
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}  # 构建索引时,丢弃20%的低权重值(减少存储,噪声, 精度影响较小)
            )

            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    schema=schema,
                    index_params=index_params
                )
            except Exception as e:
                logger.error(f'创建集合 {self.collection} 失败: {e}')

            logger.info(f'集合 {self.collection} 创建成功')

        else:
            logger.info(f'集合 {self.collection} 已存在')

        self.client.load_collection(self.collection)


    def insert_data(self, documents):
        texts = [doc.page_content for doc in documents]
        # print(len(texts))
        # print(texts)

        embedding =self.embedding_function(texts)
        # print(embedding['sparse'])

        data = []
        for i, doc in enumerate(documents):
            hash_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

            sparse_vector = {}

            row = embedding['sparse'][[i]]
            # print(row)

            indices = row.indices   # 非零元素索引
            values = row.data       # 权重

            for index, value in zip(indices, values):
                sparse_vector[index] = value
                # print(sparse_vector)

            data.append(
                {
                    'id': hash_id,
                    'text': doc.page_content,
                    'dense_vector': embedding['dense'][i],
                    'sparse_vector': sparse_vector,
                    'parent_id': doc.metadata['parent_id'],
                    'parent_content': doc.metadata['parent_content'],
                    'q_type': doc.metadata.get('q_type', 'Unknown'),
                    'timestamp': doc.metadata.get('timestamp', 'Unknown')
                }
            )

        if data:
            try:
                self.client.upsert(collection_name=self.collection, data=data)
                logger.info(f'Milvus数据插入成功')
            except pymilvus.ExceptionsMessage as e:
                logger.error(f'Milvus数据插入失败: {e}')



    def hybrid_retrieval(self, query, top_k=Config().RETRIEVAL_K, q_type=None):
        embedding = self.embedding_function([query])
        query_dense_vector = embedding['dense'][0]
        query_sparse_vector = {}

        row = embedding['sparse'][[0]]
        indices = row.indices
        values = row.data

        for index, value in zip(indices, values):
            query_sparse_vector[index] = value

        file_expr = f'q_type == "{q_type}"' if q_type else ''
        try:
            dense_request = AnnSearchRequest(
                data=[query_dense_vector],
                anns_field='dense_vector',
                param={'metric_type': 'IP', 'params': {'nprobe': 10}},
                limit=top_k,
                expr=file_expr,
            )
            sparse_request = AnnSearchRequest(
                data=[query_sparse_vector],
                anns_field='sparse_vector',
                param={'metric_type': 'IP', 'params': {}},
                limit=top_k,
                expr=file_expr,
            )

            reranker = WeightedRanker(0.7, 1.0)

            results = self.client.hybrid_search(
                collection_name=self.collection,
                reqs=[dense_request, sparse_request],
                output_fields=['text', 'parent_id', 'parent_content', 'q_type', 'timestamp'],
                ranker=reranker,
                limit=top_k,
            )[0]

            # 统一格式：Document 对象列表，方便后续处理
            sub_chunks = [self.__doc_from_hit(hit['entity']) for hit in results]

            # 去重
            unique_parent_chunks = self.__get_unique_parent_chunk(sub_chunks)

            if len(unique_parent_chunks) < 2:
                return unique_parent_chunks[:Config().CANDIDATE_M]

            if unique_parent_chunks:
                pairs = [[query, doc.page_content] for doc in unique_parent_chunks]
                scores = self.reranker.predict(pairs)
                ranked_chunks = [doc for _, doc in sorted(zip(scores, unique_parent_chunks), reverse=True)]
            else:
                ranked_chunks = []

            return ranked_chunks[:Config().CANDIDATE_M]

        except Exception as e:
            logger.error(f'混合检索失败: {e}')


    def __doc_from_hit(self, hit):
        return Document(
            page_content=hit['text'],
            metadata={
                'parent_id': hit['parent_id'],
                'parent_content': hit['parent_content'],
                'q_type': hit['q_type'],
                'timestamp': hit['timestamp']
            }
        )

    def __get_unique_parent_chunk(self, sub_chunks):
        unique_parent_chunks = []
        seen_parent_ids = set()
        for sub_chunk in sub_chunks:
            parent_content = sub_chunk.metadata.get('parent_content', sub_chunk.page_content)
            if sub_chunk.metadata['parent_id'] not in seen_parent_ids:
                unique_parent_chunks.append(Document(page_content=parent_content, metadata=sub_chunk.metadata))
                seen_parent_ids.add(sub_chunk.metadata['parent_id'])
        return unique_parent_chunks




if __name__ == '__main__':
    vector_store = VectorStore()
    # data_path = '../data/亚健康_data'
    # documents = process_documents(data_path)
    # vector_store.insert_data(documents)
    result = vector_store.hybrid_retrieval('什么是亚健康？',q_type='亚健康')
    print(result)