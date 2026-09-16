import configparser
import os

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir_path)

config_file_path = os.path.join(project_root, 'config.ini')


class Config:
    def __init__(self, config_file=config_file_path):
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag/models')
        self.DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag/document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        self.config.read(config_file, encoding='utf-8')

        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='health_qa'))

        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME', self.config.get('milvus', 'database_name', fallback='iHealth'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', self.config.get('milvus', 'collection_name', fallback='health_rag'))

        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key'))
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url', fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        self.VALID_SOURCES = eval(self.config.get('app', 'valid_sources', fallback='["亚健康", "体重管理", "作息睡眠", "心理健康", "营养学", "运动健身"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='182****1901')

        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')

        self.HISTORY_LENGTH = self.config.getint('app', 'history_length', fallback=5)

config = Config()
