from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState
import os
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional
import time
import re
from base import logger
from main_v2 import IntegratedSystem


# 1. 创建FastAPI应用实例 -> 自动生成Swagger在线文档(/docs).
app = FastAPI(
    title="i-Health智能问答系统API",
    description="融合MySQL(BM25检索)与RAG(Milvus向量检索+DashScope大模型)的医疗健康智能问答服务, "
                "支持Http非流式问答与WebSocket流式打字机输出、会话历史管理、知识库类型过滤.",
)

# 2. 全局注入CORS跨域中间件 -> 配置前后端跨域访问规则.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有前端来源访问, 生产环境需配置为具体前端域名, 避免安全风险.
    allow_credentials=True,     # 允许携带Cookie / Token 等身份凭证.
    allow_methods=["*"],        # 允许所有HTTP方法: GET / POST / PUT / DELETE ...
    allow_headers=["*"],        # 允许客户端携带任意自定义请求头.
)

# 3. 创建静态资源存储目录 static -> exist_ok=True表示目录已存在时不抛异常.
os.makedirs('static', exist_ok=True)

# 4. 全局实例化 i-Health 问答核心系统 (启动时完成: MySQL连接、Redis连接、BM25建模、向量库连接、LLM初始化).
qa_system = IntegratedSystem()


GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是i-Health智能健康助手，很高兴为你提供饮食营养、体重管理、作息睡眠、运动健身、亚健康与心理健康等方面的咨询服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是i-Health智能健康助手，专注于健康养生与疾病预防知识的解答，有任何健康问题都可以随时问我！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是i-Health智能健康助手，随时为你解答健康相关问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为你解答饮食、运动、睡眠等健康问题！有什么我可以帮你的？"
    }
]


class QueryRequest(BaseModel):
    query: str                           # 查询内容, 必填.
    source_filter: Optional[str] = None  # 知识库类型过滤(如: 营养学/体重管理/亚健康...), 可选.
    session_id: Optional[str] = None     # 会话ID, 可选; 不传则由服务端生成新会话.


class QueryResponse(BaseModel):
    answer: str            # 最终回复内容.
    is_streaming: bool     # 是否需要流式输出 -> True表示需要切换到WebSocket接口接收逐字答案.
    session_id: str        # 会话ID -> 用于上下文历史关联.
    processing_time: float # 处理耗时 -> 本次问答完整处理时间(秒).


# 1. 将本地 static 目录挂载至 /static 路径 -> 前端页面所需的css/js等静态资源.
app.mount('/static', StaticFiles(directory='static'), name='static')

# 2. 根路径GET接口 -> 访问首页HTML页面(static/index.html), 未放置前端文件时给出服务说明.
#    注意: FileResponse 默认不带 Cache-Control, 浏览器会按启发式规则长期复用缓存副本,
#    导致前端改版后用户仍看到旧页面 -> 这里显式禁用HTML缓存, 保证刷新即可拿到最新页面.
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
async def read_root():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html')
    if os.path.exists(index_path):
        # 存在首页 -> 直接返回前端页面(禁用缓存).
        return FileResponse(index_path, headers=NO_CACHE_HEADERS)
    # 尚未放置前端 -> 返回服务健康说明, 方便接口调试.
    return {
        "app": "i-Health 智能问答系统",
        "message": "static/index.html 尚未就绪, 请将前端页面放入 static 目录后刷新访问",
        "docs": "/docs",
        "health": "/health",
        "sources": "/api/sources",
        "sessions": "/api/sessions",
        "create_session": "/api/create_session",
    }


@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())          # 生成唯一会话ID.
    logger.info(f"创建新会话: {session_id}")
    return {"session_id": session_id}       # 返回会话ID.


@app.get("/api/sessions")
async def list_sessions(limit: int = 50):
    try:
        # 数据库查询为阻塞操作 -> 放到线程池执行, 避免阻塞事件循环.
        sessions = await asyncio.to_thread(qa_system.list_conversations, limit)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录 -> 内部存储结构为 [{"query":.., "answer":..}, ...]
        raw_history = await asyncio.to_thread(qa_system.get_history_conversation, session_id)
        # 转换为前端约定格式 [{"question":.., "answer":..}, ...]
        history = [
            {"question": item.get("query", ""), "answer": item.get("answer", "")}
            for item in raw_history
        ]
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"获取历史记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = await asyncio.to_thread(qa_system.delete_conversation, session_id)
    if success:
        logger.info(f"会话 {session_id} 历史记录已清除")
        return {"status": "success", "message": "历史记录已清除"}
    logger.error(f"清除会话 {session_id} 历史记录失败")
    raise HTTPException(status_code=500, detail="清除历史记录失败")


@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除首尾空格.
    for pattern_info in GREETING_PATTERNS:
        # 正则匹配, 忽略大小写 -> 兼容中英文大小写混用场景.
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]  # 命中则返回预设回复.
    return None  # 未命中返回 None.


def _resolve_q_type(source_filter: Optional[str]) -> Optional[str]:
    if source_filter and source_filter in qa_system.config.VALID_SOURCES:
        return source_filter  # 合法类型 -> 按该类型过滤检索.
    if source_filter:
        logger.warning(f"无效的知识库类型: {source_filter}, 本次查询将不进行类型过滤")
    return None  # 非法/为空 -> 不过滤.


def _save_history(session_id: Optional[str], query: str, answer: str):
    if not session_id or not answer:
        return
    try:
        qa_system.update_history_conversation(session_id, query, answer)
    except Exception as e:
        logger.error(f"保存会话历史失败: {e}")


# 流程: 问候语直接返回 -> BM25在MySQL问答库命中(置信度>=0.85)直接返回答案
#       -> 未命中则提示前端切换WebSocket接口进行流式RAG问答.
@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()  # 记录开始时间.
    # 使用请求携带的session_id或生成新会话ID.
    session_id = request.session_id or str(uuid.uuid4())

    # 1. 检查是否为日常问候 -> 命中则直接返回预设回复.
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 问候语同样落库 -> 保证该会话能被"历史会话"列表持久化展示.
        await asyncio.to_thread(_save_history, session_id, request.query, greeting_response)
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": round(time.time() - start_time, 3),
        }

    # 2. BM25 在 MySQL 问答库中检索 -> (匹配答案, 是否需要走RAG流程).
    #    检索为阻塞式CPU/IO操作 -> 放到线程池执行.
    answer, need_rag = await asyncio.to_thread(qa_system.bm25_search, request.query, 0.85)
    if answer:
        # 2.1 MySQL问答库命中 -> 直接返回答案, 无需流式; 同时写入会话历史.
        await asyncio.to_thread(_save_history, session_id, request.query, answer)
        logger.info(f"MySQL问答库命中, 直接返回答案, 耗时: {time.time() - start_time:.3f}s")
        return {
            "answer": answer,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": round(time.time() - start_time, 3),
        }
    if need_rag:
        # 2.2 MySQL未命中 -> 需要走RAG检索生成 -> 提示前端建立WebSocket连接获取流式答案.
        logger.info(f"MySQL未命中, 需要走RAG流式问答: {request.query}")
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": round(time.time() - start_time, 3),
        }
    # 2.3 非法查询(空串等) -> 返回友好提示.
    return {
        "answer": "您输入的问题为空或无法处理, 请重新描述您的健康问题",
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": round(time.time() - start_time, 3),
    }


_STREAM_DONE = object()  # 工作线程结束哨兵(对象身份比较, 不与任何事件冲突)


# 条件发送工具 -> 连接已断开时不再尝试推送, 避免抛异常打断清理流程.
async def _ws_send(websocket: WebSocket, payload: dict):
    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json(payload)


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受 WebSocket 连接.
    loop = asyncio.get_running_loop()  # 当前事件循环 -> 供工作线程回投事件.
    try:
        while True:
            # 1. 接收客户端消息并解析.
            data = await websocket.receive_text()
            request_data = json.loads(data)

            query = (request_data.get("query") or "").strip()    # 用户问题.
            source_filter = request_data.get("source_filter")     # 知识库类型过滤(可选).
            session_id = request_data.get("session_id") or str(uuid.uuid4())  # 会话ID.
            q_type = _resolve_q_type(source_filter)               # 校验并转换为q_type过滤参数.
            start_time = time.time()  # 记录开始时间.

            # 2. 发送流式开始标志.
            await _ws_send(websocket, {"type": "start", "session_id": session_id})

            # 3. 检查是否为日常问候 -> 直接推送预设回复后结束本轮.
            greeting_response = check_greeting(query)
            if greeting_response:
                await _ws_send(websocket, {
                    "type": "token",
                    "token": greeting_response,
                    "session_id": session_id,
                })
                # 问候语同样落库 -> 保证该会话出现在"历史会话"列表中.
                await asyncio.to_thread(_save_history, session_id, query, greeting_response)
                # 发送流式结束标志.
                await _ws_send(websocket, {
                    "type": "end",
                    "session_id": session_id,
                    "is_complete": True,
                    "processing_time": round(time.time() - start_time, 3),
                })
                break  # 问候语处理完毕, 关闭连接.

            # 4. 阻塞式问答核心(BM25/向量检索/大模型调用)放入独立工作线程执行:
            #    - 旧实现直接在事件循环中迭代同步生成器, 会把整个事件循环卡死 -> 期间任何消息都发不出去;
            #    - 现在线程内逐事件产出, 通过线程安全队列回投事件循环, 边生成边下发.
            event_queue: asyncio.Queue = asyncio.Queue()

            def _emit(event):
                try:
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
                except RuntimeError:
                    # 事件循环已关闭(客户端断开且服务收尾) -> 丢弃事件即可.
                    pass

            def _produce():
                try:
                    for event in qa_system.query_stream(
                        query=query,
                        q_type=q_type,
                        conversation_id=session_id,
                    ):
                        _emit(event)
                except Exception as exc:
                    logger.error(f"流式问答执行异常: {exc}", exc_info=True)
                    _emit({"type": "error", "error": str(exc)})
                finally:
                    _emit(_STREAM_DONE)  # 无论成功失败都通知主协程收尾.

            worker = asyncio.create_task(asyncio.to_thread(_produce))

            # 5. 逐事件推送: 队列 await 会让出事件循环, token 一产出即下发.
            while True:
                event = await event_queue.get()
                if event is _STREAM_DONE:
                    break

                event_type = event.get("type")
                if event_type == "status":
                    # 阶段状态 -> 前端展示"正在检索知识库..."等提示.
                    await _ws_send(websocket, {
                        "type": "status",
                        "message": event.get("message", ""),
                        "session_id": session_id,
                    })
                elif event_type == "token":
                    token = event.get("token", "")
                    if token:
                        await _ws_send(websocket, {
                            "type": "token",
                            "token": token,
                            "session_id": session_id,
                        })
                elif event_type == "error":
                    logger.error(f"会话 {session_id} 流式回答出错: {event.get('error')}")
                    await _ws_send(websocket, {
                        "type": "error",
                        "error": event.get("error", "未知错误"),
                        "session_id": session_id,
                    })
                elif event_type == "end":
                    await _ws_send(websocket, {
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": round(time.time() - start_time, 3),
                    })
                    logger.info(f"会话 {session_id} 流式回答完成, 耗时: {time.time() - start_time:.2f}s")

            await worker  # 等待工作线程收尾(会话历史此时已落库).
            break  # 一轮问答结束 -> 关闭连接(前端每轮问答新建一个连接).

    except WebSocketDisconnect as e:
        # 客户端主动断开 -> 记录断开信息.
        logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 服务端异常 -> 记录错误并尝试推送给客户端.
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await _ws_send(websocket, {"type": "error", "error": str(e)})
        except Exception:
            pass  # 推送失败说明连接已不可用, 忽略即可.
    finally:
        # 收尾: 关闭WebSocket连接(若仍处于连接状态).
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception as e:
            logger.error(f"关闭WebSocket连接出错: {str(e)}")


if __name__ == "__main__":
    import uvicorn  # uvicorn: 异步Web服务容器, 负责监听端口、处理高并发请求.

    # 从环境变量读取主机与端口, 默认监听 0.0.0.0:8080.
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8081))

    # reload=False 关闭热重载, 生产环境禁用以避免性能损耗与重复初始化.
    uvicorn.run("app:app", host=host, port=port, reload=False)
