import warnings
warnings.filterwarnings("ignore")

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from base import logger, Config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


# 'ollama'：使用本地 Ollama 模型评估
# 'api'   ：调用外部 API 模型评估
EVAL_BACKEND = 'api'

# 本地 Ollama 相关配置
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_LLM_MODEL = 'qwen2.5:7b'
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large:latest'

API_EMBEDDING_MODEL = 'text-embedding-v4'

# ragas 运行时参数：单个任务默认 180s 超时，本地 7B 模型不够用，超时会被 ragas 吞成 NaN
EVAL_RUN_CONFIG = RunConfig(timeout=900, max_retries=1, max_workers=4)


class RagasChatOllama(ChatOllama):
    """兼容 ragas 0.2.6 与 langchain-ollama 1.x。

    ragas 0.2.6 会把 temperature 作为参数传给 ChatOllama.agenerate_prompt()，
    而 langchain-ollama 1.x 会把它原样透传给 ollama 客户端，抛出
    TypeError: AsyncClient.chat() got an unexpected keyword argument 'temperature'。
    该异常会被 ragas 静默吞掉（raise_exceptions 默认为 False），导致所有指标全是 NaN。
    这里丢弃这个参数，温度改由构造函数设置（评测场景用 0 最稳定）。
    """

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        kwargs.pop('temperature', None)
        return super().generate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        kwargs.pop('temperature', None)
        return await super().agenerate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)


def build_models():
    config = Config()

    if EVAL_BACKEND == 'ollama':
        llm = RagasChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        logger.info(f'使用本地Ollama模型评估: {OLLAMA_LLM_MODEL} ({OLLAMA_BASE_URL})')

    elif EVAL_BACKEND == 'api':
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0
        )
        embeddings = OpenAIEmbeddings(
            model=API_EMBEDDING_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,

            # 默认 True 时 langchain 会把文本切成 token 数组再发送，
            # DashScope 兼容模式只接受字符串，会报 400 InvalidParameter
            check_embedding_ctx_length=False
        )
        logger.info(f'使用外部API模型评估: {config.LLM_MODEL} ({config.DASHSCOPE_BASE_URL})')

    else:
        logger.error(f"不支持的 EVAL_BACKEND: {EVAL_BACKEND}，可选 'ollama' 或 'api'")
        raise ValueError(f"不支持的 EVAL_BACKEND: {EVAL_BACKEND}，可选 'ollama' 或 'api'")

    ragas_llm = LangchainLLMWrapper(langchain_llm=llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings=embeddings)
    return ragas_llm, ragas_embeddings


try:
    with open('./rag_evaluate_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        logger.info('文件加载成功')
except Exception as e:
    logger.error(f'文件加载失败: {e}')
    raise

eval_data = {
    'question': [item['question'] for item in data],
    'retrieved_contexts': [[item['context']] for item in data],
    'answer': [item['answer'] for item in data],
    'ground_truth': [item['ground_truth'] for item in data]
}
dataset = Dataset.from_dict(eval_data)

try:
    ragas_llm, ragas_embeddings = build_models()
    logger.info('模型加载成功')
except Exception as e:
    logger.error(f'模型加载失败: {e}')
    raise

try:
    logger.info(f'开始评估，评估方式: {EVAL_BACKEND}')
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=EVAL_RUN_CONFIG,
        raise_exceptions=True
    )
except Exception as e:
    logger.error(f'评估失败: {e}')
    raise

print(f'评估结果：{result}')
logger.info(f'评估结果：{result}')
