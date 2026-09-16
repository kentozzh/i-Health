import warnings
warnings.filterwarnings("ignore")

from mysql import RedisClient, MySQLClient, BM25Search
from rag import VectorStore, RAGSystem, LLMClient
from base import logger, Config
import time

class IntegratedSystem:
    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25 = BM25Search(redis_client=self.redis_client, mysql_client=self.mysql_client)
        self.llm_client = LLMClient()
        self.call_llm = LLMClient().call_llm
        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(vector_store=self.vector_store, llm=self.call_llm)


    def query(self, query, q_type=None):
        start_time = time.time()
        self.logger.info(f'开始处理查询：{query}, 问题类型：{q_type}')

        answer, require_rag = self.bm25.search(query=query)

        if answer:
            self.logger.info('bm25查询成功，直接返回结果')
            process_time = time.time() - start_time
            self.logger.info(f'查询用时：{process_time}')
            return answer
        elif require_rag:
            self.logger.info('正在准备执行RAG....')
            answer = self.rag_system.final_answer(query=query, query_type=q_type)
            process_time = time.time() - start_time
            self.logger.info(f'RAG处理完成，用时：{process_time}')
            return answer
        else:
            self.logger.warning('输入为空或不合法...')
            return '请输入合法查询！'


def main():
    system = IntegratedSystem()
    system.logger.info('集成系统初始化完成')

    try:
        print('❤️'*15)
        print(f'请输入您的问题\n该系统支持以下查询类型：{"./".join(Config().VALID_SOURCES).strip()}')
        print('输入"exit"退出系统')

        while True:
            query = input('\n您的提问：')

            if query.lower() == 'exit':
                system.logger.info('用户输入"exit"，系统退出')
                print('感谢使用i-Health，再见！')
                break

            query_type = input('请输入过滤来源, 直接默认不过滤：').strip()
            if not query_type or query_type not in Config().VALID_SOURCES:
                system.logger.info('用户输入非法过滤条件，不进行过滤')
                print('无效过滤来源！本次查询将不进行过滤...')
                query_type = None

            answer = system.query(query=query, q_type=query_type)
            print(f'查询结果：\n {answer}')

    except Exception as e:
        system.logger.info('集成系统异常')
        print(f'系统出错，请联系管理员处理：{Config().CUSTOMER_SERVICE_PHONE}')
        raise

    finally:
        system.mysql_client.close()
        system.logger.info('MySQL连接关闭....')

if __name__ == '__main__':
    main()