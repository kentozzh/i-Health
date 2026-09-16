from base import logger
from mysql import MySQLClient
from mysql import RedisClient
from mysql import BM25Search

import time

class MySQLSystem:
    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25 = BM25Search(mysql_client=self.mysql_client, redis_client=self.redis_client)


    def search(self, query):
        start = time.time()
        answer, _ = self.bm25.search(query)
        if answer:
            logger.info("MySQL查询成功")
        else:
            logger.info("MySQL查询失败")
            answer = 'MySQL中没有找到相关结果'

        process_time = time.time() - start
        logger.info(f"查询耗时: {process_time}秒")
        return answer



def main():
    mysql_system = MySQLSystem()
    try:
        print("\n欢迎使用MySQL查询系统")
        print("输入 'exit' 退出系统\n")
        while True:
            query = input("输入您的问题：\n")
            if query.lower() == 'exit':
                print('感谢您的使用，我们下次再见！')
                break
            answer = mysql_system.search(query)
            print(answer)

    except Exception as e:
        logger.error(f'MySQL系统错误：f{e}')

    finally:
        mysql_system.mysql_client.close()

if __name__ == '__main__':
    main()