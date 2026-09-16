# i-Health RAG 智能健康问答系统

面向饮食营养 / 体重管理 / 作息睡眠 / 运动健身 / 亚健康 / 心理健康六大场景的
垂直领域智能问答系统。

## 核心特性
- 四级递进召回：问候语规则 → Redis → BM25 标准问答库 → RAG 生成
- BERT 意图分流：通用问题跳过检索直答
- 混合检索：BGE-M3 稠密+稀疏双向量 → WeightedRanker 融合 → bge-reranker-large 精排
- 父子分块：子块检索保证命中精度，回溯父块保证上下文完整
- Query 改写策略路由：HyDE / 子查询 / 回溯简化 / 直接检索
- 真流式输出：WebSocket 逐 token 下发 + 阶段进度提示
- 多格式文档入库：PDF / DOCX / PPT / 图片，OCR 自动降级

## 评测结果 (RAGAS, 30 条评测集)
| 指标              | 分数 |
| ----------------- | ---- |
| faithfulness      | 0.84 |
| answer_relevancy  | 0.74 |
| context_precision | 1.00 |
| context_recall    | 0.94 |

## 架构图

![i-Health 总体架构](docs/architecture.svg)

> 上图源文件为 [`docs/architecture.drawio`](docs/architecture.drawio)，可用 [draw.io](https://app.diagrams.net) 打开编辑；
> 若所用 Markdown 阅读器不渲染 SVG，可直接查看 [`docs/architecture.png`](docs/architecture.png)。

## 环境依赖
MySQL 8.0 / Redis / Milvus 2.5 需本地启动

## 快速开始
1. `pip install -r requirements.txt`（推荐，仅安装直接依赖）
   如需与开发环境完全一致的版本：`pip install -r requirements-lock.txt`
2. `cp config.example.ini config.ini` 并填入 DashScope API Key
3. 下载模型（见下）
4. 初始化数据（建表 → 导入 FAQ → 文档向量入库）：
   ```bash
   python init_data.py            # 幂等初始化；已存在则跳过 FAQ 导入
   python init_data.py --force    # 强制重建（清空 FAQ 表 + 重建 Milvus 集合）
   python init_data.py --only mysql    # 只初始化问答库（不加载大模型，很快）
   ```
5. `python app.py` → http://localhost:8080

## 模型下载
本项目不包含模型权重（约 7.5GB），请自行下载到 `rag/models/`：
| 模型 | 用途 | 来源 |
| --- | --- | --- |
| BAAI/bge-m3 | 稠密+稀疏向量 | https://huggingface.co/BAAI/bge-m3 |
| BAAI/bge-reranker-large | 交叉编码器重排 | https://huggingface.co/BAAI/bge-reranker-large |
| google-bert/bert-base-chinese | 意图分类基座 | https://huggingface.co/google-bert/bert-base-chinese |
| damo/nlp_bert_document-segmentation-chinese-base | 文档语义分段（**可选**） | https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base |

> 最后一项来自**魔搭 ModelScope**（阿里达摩院官方模型），仅在启用语义分段时才需要；
> 缺失时 `rag/text_splitter/__init__.py` 的条件导入会自动跳过，不影响主流程。

`bert_intent_recognizer`（意图分类权重）**首次运行会自动用
`rag/bert_train_data/` 下的 500 条语料训练并保存**，无需手动准备。

## 项目结构

```
i-Health_system/
├── app.py                          # ★ 服务入口: FastAPI HTTP 接口 + WebSocket 流式问答
├── init_data.py                    # ★ 数据初始化: 建表 -> 导入 FAQ -> 文档向量入库
├── main_v2.py                      # ★ 集成核心: 四级递进召回调度 + 会话历史管理
├── main.py                         #   早期命令行版入口 (保留作参照)
├── config.example.ini              #   配置模板 -> 复制为 config.ini 并填入 API Key
├── config.ini                      #   本地配置(含密钥) -> 已被 .gitignore 排除
├── requirements.txt                #   直接依赖 (33 个, 带分区注释)
├── requirements-lock.txt           #   完整锁定版 (222 个包, 含传递依赖)
├── LICENSE                         #   MIT 协议
├── .gitignore
│
├── docs/                           # 文档与配图
│   ├── architecture.drawio         #   架构图源文件 (用 draw.io 打开编辑)
│   ├── architecture.svg            #   架构图 (README 引用, 矢量)
│   └── architecture.png            #   架构图位图版 (备选)
│
├── base/                           # 基础设施
│   ├── config.py                   #   配置加载: config.ini + 环境变量覆盖
│   └── logger.py                   #   日志: 控制台 + 文件双通道, 10MB×5 轮转
│
├── mysql/                          # 结构化数据与缓存层
│   ├── database/mysql_client.py    #   FAQ 问答库读写 (health_qa 表)
│   ├── cache/redis_client.py       #   Redis 缓存读写
│   ├── retrieval/bm25_search.py    # ★ BM25 检索 + softmax 置信阈值(0.85) + 命中回写缓存
│   ├── utils/text_preprocess.py    #   jieba 中文分词
│   ├── main.py                     #   本层命令行入口, 可交互试查 FAQ 库
│   └── data/health_qa.csv          #   200 条精标 FAQ (8 类)
│
├── rag/                            # 检索增强生成层
│   ├── core/
│   │   ├── vector_store.py             # ★ BGE-M3 双向量 + Milvus 混合检索 + Cross-Encoder 重排
│   │   ├── rag_system_v2.py            # ★ 流式 RAG 主流程 (策略路由 + 阶段进度上报)
│   │   ├── search_strategy_selector.py #   LLM 检索策略路由 (HyDE/子查询/回溯/直接)
│   │   ├── intent_recognizer.py        # ★ BERT 通用-专业意图分类 (缺失权重时自动训练)
│   │   ├── document_preprocessor.py    #   多格式加载 + 父子分块
│   │   ├── llm_client.py               #   DashScope 同步 / 流式调用封装
│   │   ├── prompts.py                  #   Prompt 模板集中管理
│   │   └── rag_system.py               #   早期非流式版本 (保留作参照)
│   ├── document_loaders/           #   多格式文档加载器 (缺依赖时条件导入跳过)
│   │   ├── ocr.py                  #     RapidOCR 封装 (Paddle GPU / ONNX CPU 双引擎降级)
│   │   ├── pdf_loader.py            #     PDF: PyMuPDF 抽文本 + 图片 OCR + 页面旋转校正
│   │   ├── docx_loader.py            #     DOCX 解析
│   │   ├── ppt_loader.py            #     PPT / PPTX 解析
│   │   └── image_loader.py            #     图片 OCR
│   ├── text_splitter/               #   中文文本切分
│   │   ├── chinese_recursive_splitter.py  # ★ 中文递归分割器 (标点分隔优先级)
│   │   └── semantic_splitter.py   #     基于 modelscope 的语义分段 (可选依赖)
│   ├── rag_assessment/
│   │   ├── rag_as.py                   # ★ RAGAS 四维评测脚本 (Ollama / DashScope 双后端)
│   │   └── rag_evaluate_data.json      #   30 条评测集
│   ├── bert_train_data/
│   │   └── bert专业通用问题分类500条.json  # 意图分类训练语料
│   ├── main.py                         # ★ 双模式入口: --data-processing 文档入库 / 命令行问答
│   ├── data/                           # 知识库源文档 (6 个领域 / 24 篇 PDF)
│   │   ├── 营养学_data/       (15 篇)
│   │   ├── 体重管理_data/      (3 篇)
│   │   ├── 运动健身_data/      (3 篇)
│   │   ├── 亚健康_data/        (1 篇)
│   │   ├── 作息睡眠_data/      (1 篇)
│   │   └── 心理健康_data/      (1 篇)
│   └── models/                     #   模型权重(约 7.5GB) -> 已被 .gitignore 排除, 需自行下载
│
├── static/
│   └── index.html                  # 前端单页: Markdown 渲染 / 会话侧边栏 / 流式打字机
│
└── logs/                           #   运行日志 -> 已被 .gitignore 排除
```

> 说明: 上表省略各 Python 包内的常规 `__init__.py`；`rag/data/` 下文件名未逐一展开。
> 标 ★ 的文件为主要实现所在，建议从这几处开始阅读。

## License

本项目代码采用 [MIT 协议](LICENSE)，Copyright (c) 2026 张志豪。

> 注：`rag/data/` 下的知识库文档均为公开出版/发布的科普与教学资料，此处仅为技术演示用途收录，
> 版权归原作者所有；`rag/models/` 下的模型权重未随仓库分发，请按上文"模型下载"自行获取。
> 以上第三方资源均**不在本 MIT 协议覆盖范围内**。