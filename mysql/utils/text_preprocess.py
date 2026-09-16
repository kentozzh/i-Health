from base import logger
import jieba

def text_preprocess(text):
    # logger.info('开始文本预处理')      # todo 每处理一条文本时都显示，待优化
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f'文本预处理出错:{e}')
        return []


if __name__ == '__main__':
    print(text_preprocess('这是一个测试'))
