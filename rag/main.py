import os
from base import logger, Config
from rag.core.document_preprocessor import process_documents
from rag.core.vector_store import VectorStore
from rag.core.rag_system import RAGSystem
from rag.core.llm_client import LLMClient
from pymilvus import MilvusException


def main(chat_mode=True, data_path='data'):
    """
    主函数
    :param chat_mode:是否开启交互模式
        True: 交互模式
        False: 数据处理模式
    :param data_path:数据目录
    :return:无
    """

    try:
        vector_store = VectorStore()
    except MilvusException as e:
        logger.error('无法连接到向量数据库')
        return

    if chat_mode:
        client = LLMClient()

        try:
            rag_system = RAGSystem(vector_store, client.call_llm)
        except Exception as e:
            logger.error('无法创建RAG系统')
            return

        valid_sources = Config().VALID_SOURCES
        print('\n欢迎使用i-Health系统！')
        print(f'支持的查询类型：{valid_sources}')
        print('输入您要查询的问题\n或输入 "exit" 退出系统\n')

        while True:
            query = input('\n您的问题：')
            if query.lower() == 'exit':
                print('感谢使用i-Health系统！')
                break

            q_type_input = input(f'请选择查询类型（{"/".join(valid_sources)}）（直接回车默认不过滤）：').strip()
            q_type = None

            if q_type_input:
                if q_type_input in valid_sources:
                    q_type = q_type_input
                    logger.info(f'已选择查询类型：{q_type}')
                else:
                    logger.warning('无效的查询类型')
                    print(f'无效的查询类型，请输入 {"/".join(valid_sources)}，本次查询将不进行过滤')

            try:
                print('正在生成答案，请稍候...')
                answer = rag_system.final_answer(query=query, query_type=q_type)
                print(answer)
                print('\n\n\n')
            except Exception as e:
                logger.error(f'生成答案时出错：{e}')
                print('生成答案时出错，请稍后再试或联系管理员')
                return
    else:
        logger.info('当前模式为数据处理模式')
        total_chunks_added = 0
        for source_dir in Config().VALID_SOURCES:
            dir_path = os.path.join(data_path, f"{source_dir}_data")
            if os.path.exists(dir_path):
                logger.info(f"开始处理目录: {dir_path}")
                try:
                    chunks = process_documents(
                        dir_path,
                        Config().PARENT_CHUNK_SIZE,
                        Config().CHILD_CHUNK_SIZE,
                        Config().CHUNK_OVERLAP,
                    )
                    if chunks:
                        # 注: 向量入库统一入口为 init_data.py; 此处保留同一实现以便单独调试
                        vector_store.insert_data(chunks)
                        total_chunks_added += len(chunks)
                        logger.info(f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块")
                    else:
                        logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")
                except Exception as e:
                    logger.error(f"处理目录 {dir_path} 时出错: {e}")
            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")
        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='i-Health系统')
    parser.add_argument('--data-processing', action='store_true', help='数据处理模式')
    parser.add_argument('--data_path', type=str, default='data', help='数据目录')

    args = parser.parse_args()
    main(chat_mode=not args.data_processing, data_path=args.data_path)
