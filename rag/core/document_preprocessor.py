import warnings
warnings.filterwarnings("ignore")

import datetime
import os
from base import logger, Config
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from rag.document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader

from rag.text_splitter import ChineseRecursiveTextSplitter

document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader
}

def load_documents(data_path):
    documents = []
    supported_extensions = document_loaders.keys()
    try:
        logger.info('正在加载文档...')
        q_type = os.path.basename(data_path).replace("_data", "")
        # print(q_type)
        # print(os.walk(data_path))
        for root, dirs, files in os.walk(data_path):
            # print(root, '\n', dirs, '\n', files, '\n')
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                # print(file_extension)
                if file_extension in supported_extensions:
                    loader = document_loaders[file_extension]
                    if file_extension == 'txt':
                        loader = loader(file_path, encoding='utf-8')        # txt时添加编码参数
                    else:
                        loader = loader(file_path)

                    loaded_documents = loader.load()
                    # 给文件添加元数据， 方便RAG过滤检索
                    for doc in loaded_documents:
                        doc.metadata['q_type'] = q_type
                        doc.metadata['file_path'] = file_path
                        doc.metadata['timestamp'] = datetime.datetime.now().isoformat()

                    documents.extend(loaded_documents)
                    logger.info(f'成功加载文件:{file_path}')

                else:
                    logger.warning(f'不支持的文件类型:{file_extension}, 文件:{file_path}')

    except Exception as e:
        logger.error(f"加载文档时出错: {e}")

    return documents


def process_documents(
        data_path: str,
        parent_chunk_size = Config().PARENT_CHUNK_SIZE,
        child_chunk_size = Config().CHILD_CHUNK_SIZE,
        chunk_overlap = Config().CHUNK_OVERLAP
):
    documents = load_documents(data_path)
    logger.info(f'成功加载{len(documents)}个文档')

    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    child_chunks = []

    for i, doc in enumerate(documents):
        file_extension = os.path.splitext(doc.metadata.get('file_path', ''))[1].lower()

        parent_splitter = parent_splitter if file_extension != '.md' else markdown_parent_splitter
        child_splitter = child_splitter if file_extension != '.md' else markdown_child_splitter

        parent_docs = parent_splitter.split_documents([doc])
        for j, parent_doc in enumerate(parent_docs):
            parent_id = f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['content'] = parent_doc.page_content

            sub_chunks = child_splitter.split_documents([parent_doc])
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id'] = parent_id
                sub_chunk.metadata['parent_content'] = parent_doc.page_content

                sub_chunk_id = f'doc_{i}_parent_{j}_child_{k}'
                sub_chunk.metadata['sub_chunk_id'] = sub_chunk_id
                child_chunks.append(sub_chunk)

    logger.info(f'成功处理{len(child_chunks)}个子块')
    return child_chunks





if __name__ == '__main__':
    data_path = '../data/亚健康_data'
    data_path2 = '../data/运动健身_data'
    # doc = load_documents(data_path)
    # print(type(doc))
    # print(len(doc))
    # print(type(*doc))
    # print(doc)
    # print(doc[0].page_content)
    # print(doc[0].metadata)
    chunks = process_documents(data_path2)
    # for chunk in chunks:
    #     print(chunk)