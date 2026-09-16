import re
import time
import queue
import threading
from rag.core.llm_client import LLMClient
from openai import OpenAIError
from base import logger, Config
from rag.core.prompts import RAGPrompts
from rag.core.intent_recognizer import IntentRecognizer
from rag.core.search_strategy_selector import SearchStrategySelector
from rag.core.vector_store import VectorStore

import os
module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(module_path, 'models', 'bert-base-chinese')
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)



class RAGSystem_v2:
    def __init__(self, vector_store, llm, llm_stream=None):
        self.vector_store = vector_store
        self.llm = llm
        # 流式LLM调用能力(可选): 传入 LLMClient.call_llm_stream 后, final_answer_stream 才能逐token输出;
        # 未传入时自动降级为"一次性生成后再返回", 保证旧调用方式不受影响.
        self.llm_stream = llm_stream
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.intent_recognizor = IntentRecognizer()
        self.selector = SearchStrategySelector()

    # 历史对话上下文构建(流式/非流式共用)
    def _build_history_context(self, history):
        if history is not None and not isinstance(history, list):
            logger.warning("无效的历史记录格式")
            history = []
        elif history:
            history = history[-5:]
            for h in history:
                if not (isinstance(h, dict) and 'query' in h and 'answer' in h):
                    logger.warning('无效的历史记录')
                    history = []
                    break

        if not history:
            return ''

        history_context = "\n".join(
            [f"Q: {h['query']}\nA: {h['answer']}" for h in history]
        )
        logger.info(f"使用对话历史: {history_context[:100]}...")
        return history_context

    # 子查询
    def _sub_retrieval(self, query, query_type=None, progress=None):
        logger.info(f'正在进行子查询策略检索: {query} ')
        subquery_prompt = RAGPrompts.subquery_prompt()
        try:
            if progress:
                progress('正在拆解问题为多个子查询...')
            sub_text = self.llm(subquery_prompt.format(query=query))
            if not sub_text or not sub_text.strip():
                logger.warning('子查询生成为空')
                return self.vector_store.hybrid_retrieval(query, q_type=query_type) or []

            sub_text = sub_text.strip()
            if sub_text.startswith('错误:'):  # LLMClient 失败时返回错误串，不能当子查询用
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
            total = len(subquery)
            for idx, q in enumerate(subquery, 1):
                # 子查询逐个检索耗时较长 -> 逐条上报进度, 前端可持续看到进展.
                if progress:
                    progress(f'正在检索知识库(子查询 {idx}/{total}): {q[:24]}')
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
            if progress:
                progress(f'正在对 {len(unique_doc)} 个候选片段重排序...')
            pairs = [[query, doc.page_content] for doc in unique_doc]
            scores = self.vector_store.reranker.predict(pairs)
            ranked = [doc for _, doc in sorted(zip(scores, unique_doc),
                                               key=lambda x: x[0], reverse=True)]
            return ranked

        except Exception as e:
            logger.error(f'子查询失败：{e}', exc_info=True)
            return []

    def _HyDE_retrieval(self, query, query_type=None, progress=None):
        logger.info(f'正在进行HyDE检索: {query} ')
        hyde_prompt = RAGPrompts.hyde_prompt()
        try:
            if progress:
                progress('正在生成假设答案(HyDE)...')
            suppose_answer = self.llm(hyde_prompt.format(query=query)).strip()
            logger.info(f'HyDE生成问题：{suppose_answer}')
            if progress:
                progress('正在按假设答案检索知识库...')
            return self.vector_store.hybrid_retrieval(
                query=suppose_answer,
                q_type=query_type
            )

        except OpenAIError as e:
            logger.error(f'HyDE检索失败：{e}')
            return []

    def _backtracking_retrieval(self, query, query_type=None, progress=None):
        logger.info(f'正在进行回溯检索: {query} ')
        backtracking_prompt = RAGPrompts.backtracking_prompt()
        try:
            if progress:
                progress('正在回溯生成更易检索的问题...')
            sub_text = self.llm(backtracking_prompt.format(query=query)).strip()
            logger.info(f'回溯生成问题：{sub_text}')
            if progress:
                progress('正在按回溯问题检索知识库...')
            return self.vector_store.hybrid_retrieval(
                query=sub_text,
                q_type=query_type
            )

        except OpenAIError as e:
            logger.error(f'回溯检索失败：{e}')
            return []


    def final_retrieval(self, query, query_type=None, strategy=None, progress=None):
        if not strategy:
            strategy = self.selector.strategy_select(query=query)

        ranked_sub_chunks = []
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._backtracking_retrieval(query, query_type=query_type, progress=progress)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._sub_retrieval(query, query_type=query_type, progress=progress)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._HyDE_retrieval(query, query_type=query_type, progress=progress)
        else:
            if progress:
                progress('正在检索知识库(直接检索)...')
            ranked_sub_chunks = self.vector_store.hybrid_retrieval(
                query, q_type=query_type
            ) or []

        final_context_docs = (ranked_sub_chunks or [])[: Config().CANDIDATE_M]
        return final_context_docs


    # v2: 增加上下文
    def final_answer(self, query, query_type=None, history=None):
        start_time = time.time()
        logger.info(f'开始处理查询: {query}')

        history_context = self._build_history_context(history)

        query_category = self.intent_recognizor.predict_category(query)

        if query_category == "通用问题":
            logger.info("查询为通用问题, 直接调用LLM生成答案")

            prompt_input = self.rag_prompt.format(
                context="",
                history=history_context,
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
            history=history_context,
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

    # 流式生成 (v2 新增)
    def _stream_llm(self, prompt, error_message):
        """
        以流式方式调用大模型 -> 逐片段产出文本.

        :param prompt: 已构建好的完整提示词
        :param error_message: LLM调用失败时返回给用户的兜底提示
        """
        stream = None
        if self.llm_stream:
            try:
                stream = self.llm_stream(prompt)
            except Exception as e:
                logger.error(f'调用LLM流式接口失败: {e}', exc_info=True)
                stream = None

        # 没有流式能力(或调用异常)降级为一次性生成, 保证功能可用(前端表现为"一次性返回整段答案").
        if stream is None:
            try:
                answer = self.llm(prompt)
            except Exception as e:
                logger.error(f'直接调用LLM失败: {e}')
                answer = error_message
            if answer:
                yield answer
            return

        if isinstance(stream, str):
            logger.error(f'流式调用LLM失败: {stream}')
            yield error_message
            return

        try:
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                except (AttributeError, IndexError):
                    continue
                content = getattr(delta, 'content', None)
                if content:
                    yield content
        except Exception as e:
            logger.error(f'流式生成中断: {e}', exc_info=True)
            yield f"\n\n抱歉, 回答生成过程中断, 请重新提问或联系人工客服: {Config().CUSTOMER_SERVICE_PHONE}"

    def final_answer_stream(self, query, query_type=None, history=None):
        """
        流式生成最终答案 (生成器).

        产出二元组: (事件类型, 内容)
            ('status', '正在检索知识库...')  -> 阶段提示, 用于让前端在检索等待期间有反馈
            ('token', '文本片段')            -> 答案增量内容, 前端逐字渲染

        实现说明: RAG检索(子查询/HyDE/回溯)与重排序的耗时远大于生成答案本身,
        因此真正的流程跑在工作线程里, 通过队列把阶段提示与token实时回传给生成器,
        调用方(WebSocket层)可在检索过程中就收到进度, 前端不会长时间"卡住不动".
        """
        event_queue = queue.Queue()
        worker = threading.Thread(
            target=self._stream_events_worker,
            args=(event_queue, query, query_type, history),
            daemon=True,
        )
        worker.start()

        # 调用方提前停止消费(如客户端断开)时, 工作线程为守护线程, 流程结束后自行退出.
        while True:
            event = event_queue.get()
            if event is None:
                break
            yield event

    def _stream_events_worker(self, event_queue, query, query_type, history):
        """工作线程: 执行完整RAG流程, 把阶段提示与生成内容实时写入队列."""
        def progress(message):
            event_queue.put(('status', message))

        start_time = time.time()
        logger.info(f'开始流式处理查询: {query}')

        error_message = f"抱歉,处理您的问题时出错,请联系人工客服: {Config().CUSTOMER_SERVICE_PHONE}"

        try:
            history_context = self._build_history_context(history)

            progress('正在理解您的问题...')
            query_category = self.intent_recognizor.predict_category(query)

            if query_category == "通用问题":
                logger.info("查询为通用问题, 跳过知识库检索, 直接流式生成答案")

                prompt_input = self.rag_prompt.format(
                    context="",
                    history=history_context,
                    question=query,
                    phone=Config().CUSTOMER_SERVICE_PHONE
                )
                progress('正在生成回答...')
            else:
                logger.info("查询为专业咨询, 执行完整 RAG 流程")

                progress('正在选择检索策略...')
                strategy = self.selector.strategy_select(query=query)
                logger.info(f"本次检索策略: {strategy}")

                progress(f'正在检索知识库({strategy})...')
                context_docs = self.final_retrieval(
                    query, query_type=query_type, strategy=strategy, progress=progress
                )

                if context_docs:
                    context = "\n\n".join([doc.page_content for doc in context_docs])
                    logger.info(f"构建上下文完成, 包含 {len(context_docs)} 个文档块")
                    progress(f'已检索到 {len(context_docs)} 个知识片段, 正在生成回答...')
                else:
                    context = ""
                    logger.info("未检索到相关文档, 上下文为空")
                    progress('未检索到相关文档, 正在生成回答...')

                prompt_input = self.rag_prompt.format(
                    context=context,
                    history=history_context,
                    question=query,
                    phone=Config().CUSTOMER_SERVICE_PHONE
                )

            for token in self._stream_llm(prompt_input, error_message):
                event_queue.put(('token', token))

            logger.info(f"流式查询处理完成, 耗时: {time.time() - start_time:.3f}s, 查询问题: {query}")

        except Exception as e:
            logger.error(f'流式RAG流程异常: {e}', exc_info=True)
            event_queue.put(('token', error_message))

        finally:
            event_queue.put(None)  # 通知生成器结束


if __name__ == '__main__':
    vector_store = VectorStore()
    llm_client = LLMClient()
    rag_system = RAGSystem_v2(
        vector_store=vector_store,
        llm=llm_client.call_llm,
        llm_stream=llm_client.call_llm_stream,
    )
    # 流式演示: status 为阶段提示, token 为逐字输出内容.
    for event_type, content in rag_system.final_answer_stream(query="每天应该喝多少水？"):
        if event_type == 'status':
            print(f'\n[{content}]')
        else:
            print(content, end='', flush=True)
    print()
