from rank_bm25 import BM25Okapi
import numpy as np

from base import logger
from mysql.utils.text_preprocess import text_preprocess
from mysql.database.mysql_client import MySQLClient
from mysql.cache.redis_client import RedisClient

class BM25Search():
    def __init__(self, redis_client, mysql_client):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.query = None
        self.tokenized_query = None
        self.bm25 = None

        self.__load_data()

    def __load_data(self):
        self.logger.info('开始加载数据')

        cache_data_key = 'cache_query'
        cache_data_value = self.redis_client.get_data(cache_data_key)
        if not cache_data_value:
            self.logger.info('redis缓存未命中')
            self.query = self.mysql_client.get_all_query()
            if not self.query:
                self.logger.warning('mysql查询结果为空')
                return
            self.logger.info('成功从MySQL中获取数据')
        else:
            self.logger.info('成功从Redis缓存中获取数据')
            self.query = cache_data_value

        # print(self.query)
        # print(type(self.query))
        # print(type(self.query[0]))
        # print(self.query[0][0])
        # for q in self.query:
        #     print(q[0])
        # tokenized_query = text_preprocess([q[0] for q in self.query])

        self.tokenized_query = [text_preprocess(q[0]) for q in self.query]
        # print(tokenized_query)

        try:
            self.bm25 = BM25Okapi(self.tokenized_query)
            self.logger.info('成功创建BM25模型')
        except Exception as e:
            self.logger.error(f'创建BM25模型出错:{e}')



    def __softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)


    def search(self, query, threshold=0.85):
        """

        :param query: 问题
        :param threshold: 走MySQL需要满足的阈值
        :return: 返回值1：匹配的答案， 返回值2：是否需要走rag流程（True：需要， False：不需要）
        """
        if not query or not isinstance(query, str):
            self.logger.warning('这不是一个有效的提问')
            return None, False

        cache_answer = self.redis_client.get_data(query)
        if cache_answer:
            self.logger.info('成功从Redis缓存中获取数据')
            return cache_answer, False
        # return 'debug', False
        try:
            tokenized_query = text_preprocess(query)
            scores = self.bm25.get_scores(tokenized_query)
            # print(len(scores))
            softmax_scores = self.__softmax(scores)
            # print(softmax_scores)
            best_index = softmax_scores.argmax()
            best_score = softmax_scores[best_index]
            if best_score >= threshold:
                matched_query = self.query[best_index][0]
                self.logger.info(f'成功从MySQL中获取数据,BM25得分:{best_score}')
                answer = self.mysql_client.match_answer(matched_query)
                if answer:
                    self.redis_client.set_data(query, answer)
                    self.logger.info('成功将数据存入Redis缓存')
                    return answer, False
                self.logger.warning('答案获取失败')
                return None, True

            self.logger.info('MySQL中没有可靠答案，需要走RAG')
            return None, True

        except Exception as e:
            self.logger.error(f'BM25搜索MySQL时出错:{e}')
            return None, True





if __name__ == '__main__':
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)

    # print(redis_client.client.keys('*'))
    # answer, _ = bm25_search.search('test_key')
    # print(answer)

    answer, _ = bm25_search.search('蔬菜每天吃多少合适？')
    print(answer)

