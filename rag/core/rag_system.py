import re
import time
from rag.core.llm_client import LLMClient
from openai import OpenAIError
from base import logger, Config
from rag.core.prompts import RAGPrompts
from rag.core.intent_recognizer import IntentRecognizer
from rag.core.search_strategy_selector import SearchStrategySelector
from rag.core.vector_store import VectorStore

class RAGSystem:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.intent_recognizor = IntentRecognizer()
        self.selector = SearchStrategySelector()

    def _sub_retrieval(self, query, query_type=None):
        logger.info(f'正在进行子查询策略检索: {query} ')
        subquery_prompt = RAGPrompts.subquery_prompt()
        try:
            sub_text = self.llm(subquery_prompt.format(query=query))
            if not sub_text or not sub_text.strip():
                logger.warning('子查询生成为空')
                return self.vector_store.hybrid_retrieval(query, q_type=query_type) or []

            sub_text = sub_text.strip()
            if sub_text.startswith('错误:'):
                logger.error(f'子查询生成失败: {sub_text}')
                return self.vector_store.hybrid_retrieval(query, q_type=query_type) or []

            subquery = []
            for line in sub_text.split('\n'):
                q = re.sub(r'^\s*(?:\d+[.、)]|[-*•])\s*', '', line).strip()
                if q and q not in subquery:
                    subquery.append(q)
            logger.info(f'生成子查询：{subquery}')

            if not subquery:
                logger.warning('未生成子查询')
                return self.vector_store.hybrid_retrieval(query, q_type=query_type) or []

            candidates = []
            for q in subquery:
                docs = self.vector_store.hybrid_retrieval(query=q, q_type=query_type) or []
                candidates.extend(docs)
                logger.info(f'子查询：{q} 检索到：{len(docs)}条文档')

            unique_doc_dict = {}
            for doc in candidates:
                key = doc.metadata.get('parent_id') or doc.page_content
                unique_doc_dict.setdefault(key, doc)
            unique_doc = list(unique_doc_dict.values())
            logger.info(f'子查询去重得到：{len(unique_doc)}条文档')

            if len(unique_doc) < 2:
                return unique_doc
            pairs = [[query, doc.page_content] for doc in unique_doc]
            scores = self.vector_store.reranker.predict(pairs)
            ranked = [doc for _, doc in sorted(zip(scores, unique_doc),
                                               key=lambda x: x[0], reverse=True)]
            return ranked

        except Exception as e:
            logger.error(f'子查询失败：{e}', exc_info=True)
            return []

    def _HyDE_retrieval(self, query, query_type=None):
        logger.info(f'正在进行HyDE检索: {query} ')
        hyde_prompt = RAGPrompts.hyde_prompt()
        try:
            suppose_answer = self.llm(hyde_prompt.format(query=query)).strip()
            logger.info(f'HyDE生成问题：{suppose_answer}')
            return self.vector_store.hybrid_retrieval(
                query=suppose_answer,
                q_type=query_type
            )

        except OpenAIError as e:
            logger.error(f'HyDE检索失败：{e}')
            return []

    def _backtracking_retrieval(self, query, query_type=None):
        logger.info(f'正在进行回溯检索: {query} ')
        backtracking_prompt = RAGPrompts.backtracking_prompt()
        try:
            sub_text = self.llm(backtracking_prompt.format(query=query)).strip()
            logger.info(f'回溯生成问题：{sub_text}')
            return self.vector_store.hybrid_retrieval(
                query=sub_text,
                q_type=query_type
            )

        except OpenAIError as e:
            logger.error(f'回溯检索失败：{e}')
            return []


    def final_retrieval(self, query, query_type=None, strategy=None):
        if not strategy:
            strategy = self.selector.strategy_select(query=query)

        ranked_sub_chunks = []
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._backtracking_retrieval(query, query_type=query_type)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._sub_retrieval(query, query_type=query_type)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._HyDE_retrieval(query, query_type=query_type)
        else:
            ranked_sub_chunks = self.vector_store.hybrid_retrieval(
                query, q_type=query_type
            ) or []

        final_context_docs = (ranked_sub_chunks or [])[: Config().CANDIDATE_M]
        return final_context_docs

    def final_answer(self, query, query_type=None):
        start_time = time.time()
        logger.info(f'开始处理查询: {query}')

        query_category = self.intent_recognizor.predict_category(query)

        if query_category == "通用问题":
            logger.info("查询为通用问题, 直接调用LLM生成答案")

            prompt_input = self.rag_prompt.format(
                context="",
                history='',
                question=query,
                phone=Config().CUSTOMER_SERVICE_PHONE
            )

            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f'直接调用LLM失败: {e}')
                answer = f"抱歉,处理您的通用问题时出错,请联系人工客服: {Config().CUSTOMER_SERVICE_PHONE}"

            processing_time = time.time() - start_time
            logger.info(f"通用知识查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

            return answer

        logger.info("查询为专业咨询, 执行完整 RAG 流程")

        strategy = self.selector.strategy_select(query=query)

        context_docs = self.final_retrieval(query, query_type=query_type, strategy=strategy)

        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"构建上下文完成, 包含 {len(context_docs)} 个文档块")
        else:
            context = ""
            logger.info("未检索到相关文档, 上下文为空")

        prompt_input = self.rag_prompt.format(
            context=context,
            history='',
            question=query,
            phone=Config().CUSTOMER_SERVICE_PHONE
        )

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f'直接调用LLM失败: {e}')
            answer = f"抱歉,处理您的专业咨询问题时出错,请联系人工客服: {Config().CUSTOMER_SERVICE_PHONE}"

        processing_time = time.time() - start_time
        logger.info(f"查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

        return answer


if __name__ == '__main__':
    vector_store = VectorStore()
    llm = LLMClient().call_llm
    rag_system = RAGSystem(vector_store=vector_store, llm=llm)
    answer = rag_system.final_answer(query="游泳和慢跑哪个运动对心脏更好")
    print(answer)
