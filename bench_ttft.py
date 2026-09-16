# =====================================================================
# 首字节延迟 (TTFT) 基准 (bench_ttft.py)
#
#   回答"真流式改造后快了多少?"这个问题 —— 之前的日志只记录了整段生成的总耗时,
#   没有记录"首个 token 到达"的时间, 因此无法回答首屏体验的改善幅度。
#
#   本脚本直接驱动 IntegratedSystem.query_stream(), 记录:
#     - 首字节延迟 (TTFT)  = 发起查询 -> 收到第一个 token 事件
#     - 总耗时             = 发起查询 -> 收到 end 事件
#     - 阶段状态           = 用于区分走的是命中路径还是 RAG 路径
#
#   用法:
#     python bench_ttft.py                # 默认 15 条 RAG 问题 + 5 条 FAQ 命中问题
#     python bench_ttft.py --limit 8
#     python bench_ttft.py --quiet
# =====================================================================

import argparse
import json
import logging
import os
import statistics
import time

from base import Config, logger


def pctl(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def load_rag_questions(limit):
    path = os.path.join(Config().PROJECT_ROOT, 'rag', 'rag_assessment', 'rag_evaluate_data.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [x['question'].strip() for x in data][:limit]


def load_faq_questions(limit):
    import csv
    path = os.path.join(Config().PROJECT_ROOT, 'mysql', 'data', 'health_qa.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig', newline='')))
    return [r['query'].strip() for r in rows][:limit]


def measure(system, query):
    """驱动一次完整问答, 返回 (ttft, total, token数, 路径标记, 状态消息)."""
    t0 = time.perf_counter()
    ttft = None
    total = None
    n_tokens = 0
    statuses = []

    for event in system.query_stream(query=query):
        now = time.perf_counter() - t0
        etype = event.get('type')
        if etype == 'status':
            statuses.append(event.get('message', ''))
        elif etype == 'token':
            n_tokens += 1
            if ttft is None:
                ttft = now
        elif etype == 'end':
            total = now
        elif etype == 'error':
            statuses.append('ERROR: %s' % event.get('error'))

    # 路径判定: RAG 路径会先推送"正在检索知识库(<策略>)"之类的阶段提示
    rag_like = any('检索知识库' in s or '子查询' in s or '假设' in s or '重排' in s
                   for s in statuses)
    path = 'RAG' if rag_like else '命中(FAQ)'
    return ttft, total, n_tokens, path, statuses


def run(args):
    if args.quiet:
        logging.getLogger('i-Health').setLevel(logging.WARNING)

    from main_v2 import IntegratedSystem

    rag_qs = load_rag_questions(args.limit)
    faq_qs = load_faq_questions(args.faq_limit)

    print('\n' + '=' * 78)
    print('首字节延迟 (TTFT) 基准  —— 测量 IntegratedSystem.query_stream')
    print('=' * 78)
    print('RAG 问题 (评测集) : %d 条' % len(rag_qs))
    print('FAQ 问题 (命中)   : %d 条' % len(faq_qs))
    print('说明: TTFT = 发起查询 -> 第一个 token 事件; 总耗时 = 到 end 事件')

    print('\n加载集成系统 (MySQL / Redis / BM25 / Milvus / BERT)...')
    system = IntegratedSystem()
    print('加载完成, 开始测量...\n')

    rows = []
    print('%-3s %-9s %9s %9s %7s  %s' % ('#', '路径', 'TTFT(s)', '总耗时(s)', 'tokens', '问题'))
    print('-' * 78)

    for i, q in enumerate(rag_qs, 1):
        ttft, total, n, path, st = measure(system, q)
        rows.append((path, ttft, total, n))
        print('%-3d %-9s %9s %9s %7d  %s'
              % (i, path, '%.2f' % ttft if ttft else '-', '%.2f' % total if total else '-', n, q[:30]))

    for j, q in enumerate(faq_qs, 1):
        ttft, total, n, path, st = measure(system, q)
        rows.append((path, ttft, total, n))
        print('%-3s %-9s %9s %9s %7d  %s'
              % ('F%d' % j, path, '%.2f' % ttft if ttft else '-',
                 '%.2f' % total if total else '-', n, q[:30]))

    print('\n' + '=' * 78)
    print('汇总')
    print('=' * 78)
    for label in ('命中(FAQ)', 'RAG'):
        sub = [r for r in rows if r[0] == label]
        if not sub:
            continue
        tt = [r[1] for r in sub if r[1] is not None]
        to = [r[2] for r in sub if r[2] is not None]
        print('\n【%s】n=%d' % (label, len(sub)))
        if tt:
            print('   TTFT   P50=%.2fs  P95=%.2fs  min=%.2fs  max=%.2fs'
                  % (pctl(tt, .5), pctl(tt, .95), min(tt), max(tt)))
        if to:
            print('   总耗时 P50=%.2fs  P95=%.2fs  min=%.2fs  max=%.2fs'
                  % (pctl(to, .5), pctl(to, .95), min(to), max(to)))
        if tt and to:
            print('   首字节占全程比例 P50=%.0f%%'
                  % (100.0 * statistics.median(tt) / statistics.median(to)))

    tot = [r[2] for r in rows if r[2] is not None]
    if tot:
        print('\n全部 %d 条: 总耗时 P50=%.1fs, P95=%.1fs' % (len(tot), pctl(tot, .5), pctl(tot, .95)))
    print('=' * 78)
    print('提示: RAG 路径的 TTFT 包含"选策略(+HyDE/子查询改写) + 向量检索 + 首块生成";')
    print('      命中路径的 TTFT 基本等于 BM25/Redis 查询耗时。')
    print()


def main():
    ap = argparse.ArgumentParser(description='首字节延迟 (TTFT) 基准')
    ap.add_argument('--limit', type=int, default=15, help='RAG 评测集取前 N 条, 默认 15')
    ap.add_argument('--faq-limit', type=int, default=5, help='FAQ 命中路径取前 N 条, 默认 5')
    ap.add_argument('--quiet', action='store_true', help='不打印检索层详细日志')
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
