import pymysql
import pandas as pd

from base import Config, logger


class MySQLClient():
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE,
            )

            self.cursor = self.connection.cursor()
            self.logger.info('mysql连接成功')
        except pymysql.MySQLError as e:
            logger.error(f'mysql连接失败：{e}')
            raise


    def create_table(self):
        mysql_sentence = """
        create table if not exists health_qa (
            id int auto_increment primary key,
            q_type varchar(20),
            query varchar(300),
            answer varchar(300)
        )
        """
        try:
            self.cursor.execute(mysql_sentence)
            self.connection.commit()
            self.logger.info("mysql表创建成功")  # todo 重复执行依旧显示
        except pymysql.MySQLError as e:
            self.logger.error(f'mysql表创建失败：{e}')
            raise

    def insert_data(self, data_path):
        try:
            data = pd.read_csv(data_path)
            self.logger.info('数据读取成功')
        except Exception as e:
            self.logger.error(f'数据读取失败：{e}')
            raise
        # print(data.head())
        for index, row in data.iterrows():
            mysql_sentence = """
            insert into health_qa (q_type, query, answer) values (%s, %s, %s)
            """
            self.cursor.execute(mysql_sentence, (row['q_type'], row['query'], row['answer']))
        try:
            self.connection.commit()
            self.logger.info('数据插入成功')
        except Exception as e:
            self.logger.error(f'数据插入失败：{e}')
            raise

    def close(self):
        try:
            self.cursor.close()
            self.connection.close()
            self.logger.info('mysql连接关闭成功')
        except pymysql.MySQLError as e:
            self.logger.error(f'mysql连接关闭失败：{e}')
            raise

    def get_all_query(self):
        mysql_sentence = """
        select query from health_qa
        """
        try:
            self.cursor.execute(mysql_sentence)
            results = self.cursor.fetchall()
            self.logger.info('获取所有问题成功')
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f'获取所有问题失败：{e}')
            return []

    def match_answer(self, query):
        mysql_sentence = """
        select answer from health_qa where query = %s
        """
        try:
            self.cursor.execute(mysql_sentence, (query,))
            result = self.cursor.fetchone()
            self.logger.info('匹配答案成功')
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error(f'匹配答案失败：{e}')
            return None


if __name__ == '__main__':
    # 测试连接
    mysql_client = MySQLClient()

    # 建表
    # mysql_client.create_table()

    # 插数据
    # data_path = '../data/health_qa.csv'
    # mysql_client.insert_data(data_path)

    # print(mysql_client.get_all_query())
    # print(mysql_client.match_answer('哪些是抗炎食物？'))
    # print(mysql_client.match_answer('哪些是抗炎食物？？？'))


    mysql_client.close()
