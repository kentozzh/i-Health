import logging
from logging.handlers import RotatingFileHandler
import os
from config import Config

module_path = os.path.abspath(__file__)
base_path = os.path.dirname(module_path)
current_path = os.path.dirname(base_path)
log_file = os.path.join(current_path, Config().LOG_FILE)


def setup_logger(log_file=log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('i-Health')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()
