import redis
from base import Config, logger
import json

class RedisClient:
    def __init__(self):
        self.logger = logger

        try:
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True           # 自动将返回的字节数据转换为字符串(避免手动解码)
            )
            self.logger.info('redis连接成功')
        except redis.RedisError as e:
            logger.error(f'redis连接失败:{e}')
            raise

    def set_data(self, key, value):
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False))     # ensure_ascii: 保证中文等非ASCII字符正常显示
            self.logger.info(f'redis set {key} 成功')
        except redis.RedisError as e:
            self.logger.error(f'redis set {key} 失败:{e}')
            raise

    def get_data(self, key):
        try:
            value = self.client.get(key)
            self.logger.info(f'redis get {key} 成功')
            return value
        except redis.RedisError as e:
            self.logger.error(f'redis get {key} 失败:{e}')
            return None

    def delete_data(self, key):
        """删除单个缓存键 -> 供数据重新导入后清理失效缓存使用(如 BM25 问答库快照)."""
        try:
            deleted = self.client.delete(key)
            self.logger.info(f'redis delete {key} 成功(删除 {deleted} 个键)')
            return bool(deleted)
        except redis.RedisError as e:
            self.logger.error(f'redis delete {key} 失败:{e}')
            return False




if __name__ == '__main__':
    # 测试连接
    redis_client = RedisClient()

    # 清除缓存数据
    # redis_client.client.flushdb()
    # print(redis_client.client.keys('*'))

    # redis_client.set_data('test_key', 'test_value')
    # print(redis_client.get_data('test_key'))
    # print(redis_client.get_data('not_exist_key'))




