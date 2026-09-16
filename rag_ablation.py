# RAG 检索策略消融实验 (rag_ablation.py) —— 实时链路版
#   用法:
#     python rag_ablation.py --limit 3              # 冒烟测试(只跑 3 题, 省钱)
#     python rag_ablation.py                        # 完整跑 3 个检索变体
#     python rag_ablation.py --modes full --candidate-m 1,2,5   # 改做 CANDIDATE_M 消融
#     python rag_ablation.py --reuse                # 复用已缓存的检索/生成结果


import argparse
import json
import os
import statistics
import time

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from base import Config, logger
from rag.core.llm_client import LLMClient
from rag.core.prompts import RAGPrompts
from rag.core.vector_store import VectorStore

# 每轮评测的度量
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']


def cache_dir():
    d = os.path.join(Config().PROJECT_ROOT, 'rag', 'rag_assessment', 'ablation_cache')
    os.makedirs(d, exist_ok=True)
    return d


def load_eval_questions(limit=None):
    """
    只取评测集的问题与标准答案; 【丢弃】其中固化的 context/answer,

    因为那正是本次实验要重新生成的对象。
    """
    path = os.path.join(Config().PROJECT_ROOT, 'rag', 'rag_assessment', 'rag_evaluate_data.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    items = [{'question': x['question'].strip(), 'ground_truth': x['ground_truth'].strip()}
             for x in data if x.get('question') and x.get('ground_truth')]
    return items[:limit] if limit else items


def build_ragas_models():
    """
    构建 RAGAS 的评判模型 (DashScope API 后端, 与 rag_as.py 的 api 分支一致).

    说明: 这里复制了 rag_as.py 中的构造逻辑而非 import —— rag_as.py 是带模块级
    副作用的脚本, 一 import 就会立刻跑一遍完整评测。
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    config = Config()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model='text-embedding-v4',
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        # 默认 True 会把文本切成 token 数组, DashScope 兼容模式只接受字符串 -> 400
        check_embedding_ctx_length=False,
    )
    return LangchainLLMWrapper(langchain_llm=llm), LangchainEmbeddingsWrapper(embeddings=embeddings)


def build_variant(vector_store, questions, mode, candidate_m, top_k, reuse):
    """对某个检索变体跑实时链路: 逐题真实检索 + 真实生成, 结果落盘缓存."""
    tag = '%s_m%d_k%d' % (mode, candidate_m, top_k)
    path = os.path.join(cache_dir(), tag + '.json')

    if reuse and os.path.exists(path):
        rows = json.load(open(path, encoding='utf-8'))
        logger.info('[%s] 复用缓存 (%d 题)' % (tag, len(rows)))
        return rows

    prompt_tpl = RAGPrompts.rag_prompt()
    llm = LLMClient()
    rows = []
    retr_ms = []

    for i, item in enumerate(questions, 1):
        q = item['question']

        t0 = time.perf_counter()
        docs = vector_store.hybrid_retrieval(q, top_k=top_k, mode=mode,
                                             candidate_m=candidate_m) or []
        retr_ms.append((time.perf_counter() - t0) * 1000.0)
        contexts = [d.page_content for d in docs]

        t1 = time.perf_counter()
        answer = llm.call_llm(prompt_tpl.format(context='\n\n'.join(contexts),
                                                history='', question=q))
        gen_s = time.perf_counter() - t1

        ctx_chars = sum(len(c) for c in contexts)
        logger.info('[%s] %2d/%d 片段=%d 上下文=%d字 检索%.0fms 生成%.1fs | %s'
                    % (tag, i, len(questions), len(contexts), ctx_chars,
                       retr_ms[-1], gen_s, q[:24]))

        rows.append({
            'question': q,
            'contexts': contexts,
            'answer': answer,
            'ground_truth': item['ground_truth'],
            'n_contexts': len(contexts),
            'context_chars': ctx_chars,
            'retrieval_ms': retr_ms[-1],
        })

    json.dump(rows, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    logger.info('[%s] 实时链路完成, 已缓存到 %s' % (tag, path))
    return rows


def evaluate_variant(rows, ragas_llm, ragas_emb, tag):
    """把某个变体的实时结果送进 RAGAS 评测."""
    dataset = Dataset.from_dict({
        'user_input': [r['question'] for r in rows],
        'response': [r['answer'] for r in rows],
        'retrieved_contexts': [r['contexts'] for r in rows],
        'reference': [r['ground_truth'] for r in rows],
    })
    logger.info('[%s] 开始 RAGAS 评测 (%d 题)' % (tag, len(rows)))
    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(timeout=900, max_retries=1, max_workers=4),
        raise_exceptions=True,
    )
    out = {}
    try:
        out = {k: float(v) for k, v in result.items() if k in METRIC_NAMES or k == 'answer_relevancy'}
    except Exception:
        pass
    # ragas 不同版本返回结构不同, 兜底再取一次
    if not out:
        try:
            df = result.to_pandas()
            for name in METRIC_NAMES:
                if name in df.columns:
                    out[name] = float(df[name].mean())
        except Exception as e:
            logger.error('[%s] 结果解析失败: %s' % (tag, e))
    logger.info('[%s] 评测结果: %s' % (tag, out))
    return out


def main():
    ap = argparse.ArgumentParser(description='RAG 检索策略消融实验 (实时链路)')
    ap.add_argument('--modes', default='dense,hybrid,full',
                    help='检索模式, 逗号分隔: dense / hybrid / full')
    ap.add_argument('--candidate-m', default='2',
                    help='喂入上下文的片段数, 可给多个做 CANDIDATE_M 消融, 如 1,2,5')
    ap.add_argument('--retrieval-k', default='3',
                    help='ANN 召回池宽度(top_k), 可给多个做 k 消融, 如 3,20')
    ap.add_argument('--limit', type=int, default=None, help='只跑前 N 题(冒烟测试用)')
    ap.add_argument('--reuse', action='store_true', help='复用已缓存的检索/生成结果')
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    ms = [int(x) for x in args.candidate_m.split(',') if x.strip()]
    ks = [int(x) for x in args.retrieval_k.split(',') if x.strip()]
    questions = load_eval_questions(args.limit)

    print('\n' + '=' * 78)
    print('RAG 检索策略消融实验 (实时链路: 真实检索 + 真实生成 + RAGAS)')
    print('=' * 78)
    print('评测题数        : %d' % len(questions))
    print('检索模式        : %s' % ', '.join(modes))
    print('候选片段数 M    : %s' % ', '.join(str(m) for m in ms))
    print('召回池宽度 K    : %s' % ', '.join(str(k) for k in ks))
    print('策略路由        : 关闭(固定直接检索, 以隔离检索方式这一单一变量)')
    print('中间结果缓存    : %s' % cache_dir())

    print('\n加载向量库与模型...')
    vector_store = VectorStore()
    ragas_llm, ragas_emb = build_ragas_models()

    results = []
    for mode in modes:
        for m in ms:
            for k in ks:
                tag = '%s_m%d_k%d' % (mode, m, k)
                rows = build_variant(vector_store, questions, mode, m, k, args.reuse)
                n_ctx = [r['n_contexts'] for r in rows]
                rms = [r['retrieval_ms'] for r in rows]
                scores = evaluate_variant(rows, ragas_llm, ragas_emb, tag)
                results.append({
                    'tag': tag, 'mode': mode, 'm': m, 'k': k, 'scores': scores,
                    'avg_ctx': statistics.mean(n_ctx) if n_ctx else 0,
                    'p50_retr_ms': statistics.median(rms) if rms else 0,
                })

    # 汇总
    print('\n' + '=' * 78)
    print('消融结果汇总  (实时检索 + 实时生成, 同一 Prompt / 同一模型 / temperature=0)')
    print('=' * 78)
    print('%-16s %8s %8s %8s %8s %10s %10s' % (
        '变体', 'faith', 'relev', 'precision', 'recall', '平均片段数', '检索P50(ms)'))
    print('-' * 78)
    for r in results:
        s = r['scores']
        print('%-16s %8s %8s %8s %8s %10.1f %10.0f' % (
            r['tag'],
            ('%.3f' % s['faithfulness']) if 'faithfulness' in s else '   -   ',
            ('%.3f' % s['answer_relevancy']) if 'answer_relevancy' in s else '   -   ',
            ('%.3f' % s['context_precision']) if 'context_precision' in s else '   -   ',
            ('%.3f' % s['context_recall']) if 'context_recall' in s else '   -   ',
            r['avg_ctx'], r['p50_retr_ms']))
    print('=' * 78)
    print('读法: 同一 K 下比 dense vs full -> 重排的净收益; 同一模式下比 K=3 vs K=20 -> 召回池宽度的影响。')
    print()

    out = os.path.join(cache_dir(), 'ablation_summary.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('汇总已保存: %s\n' % out)


if __name__ == '__main__':
    main()
