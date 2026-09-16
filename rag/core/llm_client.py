from openai import OpenAI, OpenAIError
from base import logger, Config

class LLMClient:
    def __init__(self):
        try:
            self.client = OpenAI(
                api_key=Config().DASHSCOPE_API_KEY,
                base_url=Config().DASHSCOPE_BASE_URL
            )
        except OpenAIError as e:
            logger.error('OpenAI客户端初始化失败')
            raise


    def call_llm(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": '你是一个智能助手，能严格执行prompt的指令并按要求返回可靠的结果'},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except OpenAIError as e:
            logger.error('调用LLM失败')
            return f"错误: 调用LLM失败 - {e}"

    def call_llm_stream(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": '你是一个智能助手，能严格执行prompt的指令并按要求返回可靠的结果'},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=True
            )
            return completion
        except OpenAIError as e:
            logger.error('调用LLM失败')
            return f"错误: 调用LLM失败 - {e}"
