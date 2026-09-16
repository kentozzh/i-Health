import argparse
import os
import time

from base import Config, logger
from mysql import MySQLClient, RedisClient


FAQ_CSV_RELATIVE_PATH = os.path.join('mysql', 'data', 'health_qa.csv')
# BM25 问答库快照缓存键 -> 见 mysql/retrieval/bm25_search.py 的 __load_data
BM25_CORPUS_CACHE_KEY = 'cache_query'


def _project_root():
    return Config().PROJECT_ROOT


def _faq_csv_path():
    return os.path.join(_project_root(), FAQ_CSV_RELATIVE_PATH)


def init_mysql(force=False):
    """
    创建 health_qa 表并导入 FAQ 标准问答库.

    :param force: True -> 先清空原表再重新导入; False -> 表内已有数据则跳过(幂等)
    :return: 导入完成后表内的 FAQ 条数
    """
    csv_path = _faq_csv_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'FAQ 数据文件不存在: {csv_path}')

    client = MySQLClient()
    try:
        logger.info('-' * 64)
        logger.info('步骤 1/3: 创建 FAQ 问答库表 health_qa')
        client.create_table()

        existing = len(client.get_all_query())

        logger.info(f'步骤 2/3: 导入 FAQ 数据 ({os.path.relpath(csv_path, _project_root())})')

        if existing and not force:
            logger.info(f'表内已有 {existing} 条 FAQ, 跳过导入 (如需重建请加 --force)')
            return existing

        if existing and force:
            logger.info(f'--force 生效: 清空原表 (原有 {existing} 条)')
            client.cursor.execute('truncate table health_qa')
            client.connection.commit()

        client.insert_data(csv_path)
        total = len(client.get_all_query())
        logger.info(f'FAQ 导入完成: 当前表内共 {total} 条')
        return total

    finally:
        client.close()


def invalidate_bm25_cache():
    """
    FAQ 数据变更后清理 BM25 问答库快照缓存.

    检索层启动时会把整张问答库缓存到 Redis(cache_query 键), 若不清掉,
    重新导入 FAQ 后仍会按旧快照检索。
    """
    try:
        redis_client = RedisClient()
        if redis_client.delete_data(BM25_CORPUS_CACHE_KEY):
            logger.info(f'已清理 BM25 问答库缓存键: {BM25_CORPUS_CACHE_KEY}')
        else:
            logger.info(f'BM25 问答库缓存键不存在, 无需清理: {BM25_CORPUS_CACHE_KEY}')
    except Exception as e:
        # 缓存清理失败不影响数据本身, 服务重启后也会重新加载
        logger.warning(f'清理 Redis 缓存失败(不影响本次初始化): {e}')


def _drop_milvus_collection():
    """--force 时删除整个向量集合 -> 随后 VectorStore 初始化会自动重建空集合."""
    from pymilvus import MilvusClient

    config = Config()
    client = MilvusClient(
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
        db_name=config.MILVUS_DATABASE_NAME,
    )
    try:
        if client.has_collection(config.MILVUS_COLLECTION_NAME):
            client.drop_collection(config.MILVUS_COLLECTION_NAME)
            logger.info(f'--force 生效: 已删除 Milvus 集合 {config.MILVUS_COLLECTION_NAME}')
    finally:
        client.close()


def init_milvus(data_path=None, force=False):
    """
    按知识域遍历源文档目录, 完成解析/分块/向量化/入库.

    :param data_path: 知识库根目录(其下为 <领域>_data 子目录), 默认取 config 的 DATA_DIR
    :param force: True -> 先删除 Milvus 集合再全量重建
    :return: (成功入库的领域数, 入库的子块总数)
    """
    # 延迟导入: --only mysql 时无需加载大模型与向量库依赖
    from rag import VectorStore
    from rag.core.document_preprocessor import process_documents

    config = Config()
    data_path = data_path or config.DATA_DIR

    if force:
        _drop_milvus_collection()

    logger.info('-' * 64)
    logger.info('步骤 3/3: 文档解析与向量入库')
    logger.info(f'数据目录: {data_path}')
    logger.info('首次运行需加载 BGE-M3 与 bge-reranker-large, 请耐心等待...')

    vector_store = VectorStore()

    domains = config.VALID_SOURCES
    done_domains, done_chunks = 0, 0

    for index, source in enumerate(domains, 1):
        dir_path = os.path.join(data_path, f'{source}_data')

        if not os.path.isdir(dir_path):
            logger.warning(f'[{index}/{len(domains)}] [{source}] 目录不存在, 跳过: {dir_path}')
            continue

        logger.info(f'[{index}/{len(domains)}] [{source}] 开始解析: {dir_path}')
        try:
            chunks = process_documents(
                dir_path,
                config.PARENT_CHUNK_SIZE,
                config.CHILD_CHUNK_SIZE,
                config.CHUNK_OVERLAP,
            )
        except Exception as e:
            logger.error(f'[{source}] 文档解析失败, 跳过该领域: {e}', exc_info=True)
            continue

        if not chunks:
            logger.warning(f'[{source}] 未解析出任何分块(目录为空或格式不支持), 跳过')
            continue

        try:
            vector_store.insert_data(chunks)
        except Exception as e:
            logger.error(f'[{source}] 向量入库失败, 跳过该领域: {e}', exc_info=True)
            continue

        done_domains += 1
        done_chunks += len(chunks)
        logger.info(f'[{source}] 完成: {len(chunks)} 个子块已入库')

    logger.info(f'文档入库完成: {done_domains}/{len(domains)} 个知识域, 共 {done_chunks} 个子块')
    return done_domains, done_chunks


def _milvus_row_count():
    """
    读取 Milvus 集合当前实体数, 仅用于结果展示; 失败返回 None.

    注意: 这里必须用 count(*) 聚合查询, 不能用 get_collection_stats()['row_count']。
    后者在数据尚未 flush 时会严重偏低 —— 实测插入 806 条后立刻调用只返回 55,
    手动 flush 后才变成 816(仍不等于真实值), 会给出误导性的验收结论。
    """
    try:
        from pymilvus import MilvusClient

        config = Config()
        client = MilvusClient(
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT,
            db_name=config.MILVUS_DATABASE_NAME,
        )
        try:
            if not client.has_collection(config.MILVUS_COLLECTION_NAME):
                return None
            result = client.query(
                collection_name=config.MILVUS_COLLECTION_NAME,
                output_fields=['count(*)'],
            )
            if result and isinstance(result[0], dict) and result[0]:
                return int(list(result[0].values())[0])
            return None
        finally:
            client.close()
    except Exception as e:
        logger.warning(f'读取 Milvus 集合统计失败: {e}')
        return None


def _print_summary(faq_count, chunk_info, elapsed):
    rows = _milvus_row_count()
    print('\n' + '=' * 64)
    print('i-Health 数据初始化完成')
    print('=' * 64)
    print(f'  FAQ 问答库 (MySQL health_qa) : '
          f'{faq_count if faq_count is not None else "未执行"} 条')
    if chunk_info is not None:
        print(f'  文档入库 (Milvus)            : '
              f'{chunk_info[0]} 个知识域 / {chunk_info[1]} 个子块')
    if rows is not None:
        print(f'  Milvus 集合 {Config().MILVUS_COLLECTION_NAME:<16}: {rows} 个实体')
    print(f'  总耗时                      : {elapsed:.1f}s')
    print('=' * 64)
    print('下一步: python app.py  ->  http://localhost:8080')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='i-Health 数据初始化: 建表 -> 导入 FAQ -> 文档向量入库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--force', action='store_true',
                        help='强制重建: 清空 FAQ 表并删除 Milvus 集合后重新导入')
    parser.add_argument('--only', choices=['mysql', 'milvus'], default=None,
                        help='只执行指定步骤 (默认两步都执行)')
    parser.add_argument('--data-path', default=None,
                        help='知识库源文档根目录, 默认取 config 中的 rag/data')
    args = parser.parse_args()

    start_time = time.time()
    logger.info('=' * 64)
    logger.info('i-Health 数据初始化开始')
    logger.info(f'参数: force={args.force}, only={args.only or "all"}, '
                f'data_path={args.data_path or Config().DATA_DIR}')

    faq_count, chunk_info = None, None

    try:
        if args.only in (None, 'mysql'):
            faq_count = init_mysql(force=args.force)
            invalidate_bm25_cache()

        if args.only in (None, 'milvus'):
            chunk_info = init_milvus(data_path=args.data_path, force=args.force)

    except Exception as e:
        logger.error(f'数据初始化失败: {e}', exc_info=True)
        print('\n' + '!' * 64)
        print(f'初始化失败: {e}')
        print('请检查:')
        print('  1. MySQL / Redis / Milvus 是否已启动')
        print('  2. config.ini 中的连接信息与密码是否正确')
        print('  3. rag/models/ 下模型权重是否已按 README 下载完整')
        print('!' * 64 + '\n')
        return 1

    _print_summary(faq_count, chunk_info, time.time() - start_time)
    logger.info(f'数据初始化全部完成, 总耗时 {time.time() - start_time:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
