import warnings

from rag.core.rag_system_v2 import RAGSystem_v2

warnings.filterwarnings("ignore")

from mysql import RedisClient, MySQLClient, BM25Search
from rag import VectorStore, LLMClient
from base import logger, Config
import time

import pymysql
import uuid
import threading



class IntegratedSystem:
    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25 = BM25Search(redis_client=self.redis_client, mysql_client=self.mysql_client)
        self.llm_client = LLMClient()
        self.call_llm = self.llm_client.call_llm
        # 流式LLM调用能力 -> RAG流程可逐token输出, 避免前端长时间无响应.
        self.call_llm_stream = self.llm_client.call_llm_stream
        self.vector_store = VectorStore()
        self.rag_system = RAGSystem_v2(
            vector_store=self.vector_store,
            llm=self.call_llm,
            llm_stream=self.call_llm_stream,
        )

        # pymysql 的连接/游标不是线程安全的: Web层会把阻塞式问答放到工作线程执行,
        # 这里用可重入锁串行化所有数据库操作, 避免游标被并发复用导致的协议错乱.
        self._db_lock = threading.RLock()

        self.init_conversation_table()


    def init_conversation_table(self):
        sql_sentence = """
            create table if not exists conversations (
                id int auto_increment primary key,
                conversation_id varchar(36) not null,
                query text not null,
                answer text not null,
                timestamp datetime default current_timestamp,
                index idx_conversation_id (conversation_id)
            )
        """
        try:
            with self._db_lock:
                self.mysql_client.cursor.execute(sql_sentence)
                self.mysql_client.connection.commit()
            self.logger.info("对话表初始化完成")
        except pymysql.MySQLError as e:
            self.logger.error(f"对话表初始化失败：{e}")
            raise

    def __fetch_recent_history(self, conversation_id):
        # 按自增主键倒序取最近N轮: timestamp 只精确到秒, 同一秒内的多轮问答顺序不确定, 用 id 保证顺序稳定.
        sql_sentence = """
            select query, answer from conversations
            where conversation_id = %s
            order by id desc limit %s
        """
        try:
            with self._db_lock:
                self.mysql_client.cursor.execute(sql_sentence,(conversation_id, self.config.HISTORY_LENGTH))
                history = [{'query': row[0], 'answer':row[1]} for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"对话表查询失败：{e}")
            return  []


    def get_history_conversation(self, conversation_id):
        return self.__fetch_recent_history(conversation_id)

    def list_conversations(self, limit=50):
        """
        查询会话列表 -> 供前端"历史会话"侧边栏渲染.
        每个会话返回: 会话ID、标题(该会话首轮提问)、轮数、最近更新时间.
        """
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 50

        # 标题取该会话最早的一条提问(与主流对话产品一致), last_id 用于稳定排序.
        sql_sentence = """
            select c.conversation_id,
                   (select c2.query from conversations c2
                     where c2.conversation_id = c.conversation_id
                     order by c2.id asc limit 1) as title,
                   max(c.timestamp) as last_time,
                   count(*) as rounds,
                   max(c.id) as last_id
            from conversations c
            group by c.conversation_id
            order by last_id desc
            limit %s
        """
        try:
            with self._db_lock:
                self.mysql_client.cursor.execute(sql_sentence, (limit,))
                rows = self.mysql_client.cursor.fetchall()
        except pymysql.MySQLError as e:
            self.logger.error(f"会话列表查询失败：{e}")
            return []

        conversations = []
        for conversation_id, title, last_time, rounds, _ in rows:
            conversations.append({
                'conversation_id': conversation_id,
                'title': (title or '').strip()[:50] or '新会话',
                'rounds': int(rounds or 0),
                'last_time': last_time.strftime('%Y-%m-%d %H:%M:%S') if last_time else '',
            })
        return conversations

    def update_history_conversation(self, conversation_id, query, answer):
        insert_sql_sentence = """
            insert into conversations (conversation_id, query, answer, timestamp) values (%s, %s, %s, now())
        """
        delete_sql_sentence = """
            delete from conversations where conversation_id = %s
            and id not in (select id from (select id from conversations where conversation_id = %s order by id desc limit %s) as sub)
        """
        try:
            with self._db_lock:
                self.mysql_client.cursor.execute(insert_sql_sentence, (conversation_id, query, answer))
                history = self.__fetch_recent_history(conversation_id)
                self.mysql_client.cursor.execute(delete_sql_sentence, (conversation_id, conversation_id, self.config.HISTORY_LENGTH))
                self.mysql_client.connection.commit()
            self.logger.info("对话表更新完成")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"对话表更新失败：{e}")
            with self._db_lock:
                self.mysql_client.connection.rollback()
            raise


    def delete_conversation(self, conversation_id):
        sql_sentence = """
            delete from conversations where conversation_id = %s
        """
        try:
            with self._db_lock:
                self.mysql_client.cursor.execute(sql_sentence, (conversation_id,))
                self.mysql_client.connection.commit()
            self.logger.info("对话表删除成功")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"对话表删除失败：{e}")
            with self._db_lock:
                self.mysql_client.connection.rollback()
            return False


    def bm25_search(self, query, threshold=0.85):
        """
        带锁的BM25检索 -> BM25命中时会访问MySQL游标, 与其它线程的数据库操作互斥,
        避免共享游标被并发复用(pymysql 非线程安全).
        """
        with self._db_lock:
            return self.bm25.search(query=query, threshold=threshold)

    def query_stream(self, query, q_type=None, conversation_id=None):
        """
        流式问答核心 (生成器) -> 产出结构化事件字典, 供 Web/WS 层直接转发:

            {'type': 'status', 'message': '正在检索本地知识库...'}   # 阶段提示: 让前端在检索等待期间有反馈
            {'type': 'token',  'token': '文本片段'}                  # 答案增量: 边生成边下发
            {'type': 'end',    'is_complete': True}                  # 本轮结束
            {'type': 'error',  'error': '错误信息'}                  # 本轮异常

        与旧实现的区别: RAG 分支由"等整段答案生成完再逐字符返回"改为"大模型逐token产出即刻下发",
        首字节延迟从"整个RAG流程耗时"降到"检索耗时", 前端不再长时间无响应.
        """
        start_time = time.time()
        self.logger.info(f'开始处理查询：{query}, 问题类型：{q_type}, 会话id：{conversation_id}')

        try:
            if not query or not isinstance(query, str) or not query.strip():
                yield {'type': 'token', 'token': '未找到答案'}
                yield {'type': 'end', 'is_complete': True}
                return

            history = self.get_history_conversation(conversation_id) if conversation_id else []

            yield {'type': 'status', 'message': '正在检索本地知识库...'}
            answer, require_rag = self.bm25_search(query=query)

            if answer:
                if conversation_id:
                    self.update_history_conversation(conversation_id, query, answer)
                yield {'type': 'token', 'token': answer}
                yield {'type': 'end', 'is_complete': True}
                self.logger.info(f'bm25查询成功，直接返回结果, 用时：{time.time() - start_time:.3f}s')
                return

            if not require_rag:
                self.logger.info('未找到答案')
                yield {'type': 'token', 'token': '未找到答案'}
                yield {'type': 'end', 'is_complete': True}
                return

            self.logger.info('正在准备执行RAG....')
            collected_tokens = []
            for event_type, content in self.rag_system.final_answer_stream(
                query=query, query_type=q_type, history=history
            ):
                if event_type == 'status':
                    yield {'type': 'status', 'message': content}
                else:
                    collected_tokens.append(content)
                    yield {'type': 'token', 'token': content}

            collected_answer = ''.join(collected_tokens)
            if conversation_id and collected_answer:
                self.update_history_conversation(conversation_id, query, collected_answer)

            yield {'type': 'end', 'is_complete': True}
            self.logger.info(f"查询处理耗时 {time.time() - start_time:.2f}秒")

        except Exception as e:
            # 任何未预期异常都转成事件, 避免WS连接直接断开而前端无从感知.
            self.logger.error(f'流式查询处理失败：{e}', exc_info=True)
            yield {'type': 'error', 'error': str(e)}

    def query(self, query, q_type=None, conversation_id=None):
        """
        兼容旧调用方式的流式问答接口 (生成器) -> 产出 (token, is_complete) 二元组.

        注意: 内部已改为真流式, token 会在生成过程中陆续产出, 而不是等答案生成完再一次性切分.
        """
        for event in self.query_stream(query=query, q_type=q_type, conversation_id=conversation_id):
            event_type = event.get('type')
            if event_type == 'token':
                yield event.get('token', ''), False
            elif event_type == 'end':
                yield '', True
            elif event_type == 'error':
                yield f"\n[错误] {event.get('error', '未知错误')}", True


def main():
    system = IntegratedSystem()

    conversation_id = str(uuid.uuid4())

    system.logger.info('集成系统初始化完成')

    try:
        print('❤️'*15)
        print(f'请输入您的问题\n该系统支持以下查询类型：{"./".join(Config().VALID_SOURCES).strip()}')
        print('输入"exit"退出系统')
        print(f'会话ID：{conversation_id}')

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

            answer = ''
            for token, is_complete in system.query(query=query, q_type=query_type, conversation_id=conversation_id):
                if token:
                    print(token, end='', flush=True)
                    answer += token
                if is_complete:
                    print()
                    break

            history = system.get_history_conversation(conversation_id)
            print('历史对话：')
            for idx, conversation in enumerate(history, 1):
                print(f'会话ID：{conversation_id}，会话编号：{idx}')
                print(f'问题：{conversation["query"]}')
                print(f'答案：{conversation["answer"]}')

    except Exception as e:
        system.logger.info('集成系统异常')
        print(f'系统出错，请联系管理员处理：{Config().CUSTOMER_SERVICE_PHONE}')
        raise

    finally:
        system.mysql_client.close()
        system.logger.info('MySQL连接关闭....')

if __name__ == '__main__':
    main()



