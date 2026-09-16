import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate


class RAGPrompts:
    # 直接检索
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""  
                你是一个智能助手，帮助用户回答问题。请参考用户的助手的对话历史和上下文回答问题。
                如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。  
                如果答案来源于检索到的文档，请在回答中说明。

                对话历史: {history}
                知识库: {context}  
                问题: {question}  

                回答:  
                """,
            input_variables=["context", "history", "question"],
        )

    # HyDE检索
    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            input_variables=["query"],
        )

    # 子查询
    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            input_variables=["query"],
        )

    # 回溯查询
    @staticmethod
    def backtracking_prompt():
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            input_variables=["query"],
        )

if __name__ == '__main__':
    test1 = RAGPrompts.rag_prompt().format(context='我姓张', question='我姓什么？', history='', phone='182****1901')
    print(test1)