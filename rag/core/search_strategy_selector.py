import warnings
warnings.filterwarnings("ignore")

from rag.core.llm_client import LLMClient
from langchain_core.prompts import PromptTemplate
from base import logger, Config
config = Config()


class SearchStrategySelector:
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt = self.__get_prompt()

    def __get_prompt(self):
        return PromptTemplate(
            template="""
                    你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

                    以下是几种检索增强策略及其适用场景：

                    1.  **直接检索：**
                        * 描述：对用户查询直接进行检索，不进行任何增强处理。
                        * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                            * 示例：
                                * 查询：米饭的主要成分是什么？
                                * 策略：直接检索
                    2.  **假设问题检索（HyDE）：**
                        * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                        * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                            * 示例：
                                * 查询：上班族群体需要额外补充哪些营养？
                                * 策略：假设问题检索
                    3.  **子查询检索：**
                        * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                        * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                            * 示例：
                                * 查询：比较 鸡肉 和 牛肉 在营养方面的差异。
                                * 策略：子查询检索
                    4.  **回溯问题检索：**
                        * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                        * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                            * 示例：
                                * 查询：如果一个人一整年只吃兔肉，他能活下来吗？
                                * 策略：回溯问题检索

                    根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
                    """
            ,
            input_variables=["query"],
        )



    def strategy_select(self, query):
        strategy = self.llm_client.call_llm(self.prompt.format(query=query)).strip()
        logger.info(f'query: {query} 的检索策略是: {strategy}')
        return strategy


if __name__ == '__main__':
    selector = SearchStrategySelector()
    selector.strategy_select('亚健康人群该怎么调理自己的身体？')