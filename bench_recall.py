# 四级递进召回 · 命中率基准 (bench_recall.py)
#   回答三个问题:
#     1) 各层(问候语 / Redis / BM25 / RAG)分别拦截了多少比例的请求?
#     2) BM25 置信阈值 0.85 合理吗? 扫一遍阈值, 看拦截率与正确率的权衡
#     3) 秒级 vs 毫秒级两条路径的延迟差多少?
#
#   方法要点:
#     - 分两组查询: A 组取 200 条 FAQ 原题; B 组做规则轻改写(加前缀/去标点/你->您)
#       只测原题会得到接近 100% 的虚高命中率, 没有参考价值
#     - 同时报【拦截率】(score>=阈值, 不再走 RAG) 与【正确率】(Top-1 确实是预期那条)
#       因为 bm25_search 命中后会直接返回 Top-1 的答案, 不校验是否为用户想问的问题
#     - 阈值扫描直接复用原始 softmax 分数, 不重复检索
#
#   用法:
#     python bench_recall.py                  # 默认: A 组 + B 组
#     python bench_recall.py --sets A         # 只测原题
#     python bench_recall.py --keep-cache     # 不清理 Redis 缓存(测热缓存层)
#     python bench_recall.py --limit 50       # 只跑前 50 条(快速验证)

import argparse
import csv
import json
import os
import time
from collections import Counter

from base import Config, logger
from mysql import BM25Search, MySQLClient, RedisClient
from mysql.utils.text_preprocess import text_preprocess


# 与 app.py 中的 GREETING_PATTERNS 保持一致(第 1 层)
GREETINGS = [
    '你好', '您好', 'hi', 'hello',
    '你是谁', '您是谁', '你叫什么', '你的名字', 'who are you',
    '在吗', '在不在', '有人吗',
    '干嘛呢', '你在干嘛', '做什么',
]

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ---------------------------------------------------------------------
# 负样本: 【不应该】被 FAQ 层拦截的问题 -> 用来衡量假阳性率
#   只测"应命中"的样本无法确定阈值的【下界】: 阈值扫描显示降到 0.70 拦截率仍 100%,
#   但那是在 FAQ 覆盖范围内成立; 必须用负样本证明阈值不能无限降低。
# ---------------------------------------------------------------------
NEG_DOMAIN_OUT = [
    # 明显域外: 与健康无关, 必须走 RAG(或拒答)
    '今天天气怎么样',
    '帮我写一首关于春天的诗',
    '怎么用 Python 写一个爬虫',
    '北京到上海的高铁要多久',
    '推荐几部好看的科幻电影',
    '股票什么时候买比较好',
    '怎么申请出国留学',
    '电脑蓝屏了怎么办',
]

NEG_SHORT_GENERIC = [
    # 健康域内但极短、极泛化: 最容易造成假阳性的一类
    # (与失败模式同源: 短问句与库内多条共享词元, softmax 概率质量被摊薄)
    '今天吃什么？',
    '晚饭吃什么好',
    '我想减肥',
    '我该怎么办',
    '这样正常吗',
    '有什么建议吗',
]


def load_soft_negatives():
    """
    软负样本: RAGAS 评测集里那 30 条真实风格的健康问题.

    它们是"领域内但开放/专业"的问题, FAQ 未必有对应标准答案 -> 理想应交给 RAG。
    但若恰好命中同义 FAQ 条目, 拦截也未必算错, 所以单独统计、只作参考,
    不计入假阳性率。
    """
    path = os.path.join(Config().PROJECT_ROOT, 'rag', 'rag_assessment', 'rag_evaluate_data.json')
    try:
        with open(path, encoding='utf-8') as f:
            return [item['question'].strip() for item in json.load(f) if item.get('question')]
    except Exception as e:
        logger.warning('软负样本加载失败: %s' % e)
        return []


def load_faq(limit=None):
    """从 CSV 读取 FAQ 原题 (带 q_type), 保证与入库数据同源."""
    path = os.path.join(Config().PROJECT_ROOT, 'mysql', 'data', 'health_qa.csv')
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def perturb(query):
    """规则化轻改写 -> 模拟真实用户不会照抄原题的写法."""
    q = query.strip()
    out = []
    if not q.startswith('请问'):
        out.append('请问' + q)
    stripped = q.rstrip('？?！!。.')
    if stripped != q:
        out.append(stripped)
    if '你' in q:
        out.append(q.replace('你', '您'))
    if stripped and not stripped.endswith('呢'):
        out.append(stripped + '呢')
    seen, uniq = set(), []
    for x in out:
        if x != q and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((len(s) - 1) * p)))
    return s[k]


def softmax_best(bm25, query):
    """算某条 query 对问答库的 softmax Top-1 分数与命中的 FAQ 原文."""
    tokens = text_preprocess(query)
    scores = bm25.bm25.get_scores(tokens)
    sm = bm25._BM25Search__softmax(scores)
    idx = int(sm.argmax())
    return float(sm[idx]), bm25.query[idx][0]


def run(args):
    if args.quiet:
        # 检索层每次都打日志, 大批量跑时会把报告淹掉 -> 只保留 WARNING 以上
        import logging
        logging.getLogger('i-Health').setLevel(logging.WARNING)

    faq = load_faq(args.limit)
    logger.info('第 1 层问候语规则: %d 条样本' % len(GREETINGS))

    queries = []          # (query, expected_faq_query, group)
    if 'A' in args.sets:
        for r in faq:
            queries.append((r['query'].strip(), r['query'].strip(), 'A原题'))
    if 'B' in args.sets:
        for r in faq:
            for p in perturb(r['query']):
                queries.append((p, r['query'].strip(), 'B轻改写'))

    # 负样本: 不应该被 FAQ 层拦截的问题
    hard_neg, soft_neg = [], []
    if 'N' in args.sets:
        hard_neg = NEG_DOMAIN_OUT + NEG_SHORT_GENERIC
        soft_neg = load_soft_negatives()

    print('\n' + '=' * 72)
    print('四级递进召回 · 命中率基准')
    print('=' * 72)
    print('FAQ 知识库规模        : %d 条' % len(faq))
    print('查询总数              : %d  (A原题 %d / B轻改写 %d)' % (
        len(queries),
        sum(1 for q in queries if q[2] == 'A原题'),
        sum(1 for q in queries if q[2] == 'B轻改写')))
    print('负样本                : 硬负 %d 条(域外8+短泛化6) / 软负 %d 条(评测集)'
          % (len(hard_neg), len(soft_neg)))
    print('BM25 置信阈值         : 0.85 (search 默认值)')
    print('Redis 缓存处理        : %s' % ('保留(测热缓存)' if args.keep_cache else '逐条清理(测冷缓存)'))

    mysql_client = MySQLClient()
    redis_client = RedisClient()
    try:
        bm25 = BM25Search(redis_client=redis_client, mysql_client=mysql_client)
        corpus = len(bm25.query) if bm25.query else 0
        print('BM25 语料(来自MySQL)  : %d 条' % corpus)
        if corpus == 0:
            print('\n[中止] BM25 语料为空, 请先执行 python init_data.py --only mysql')
            return

        # ---------------- 阈值扫描 ----------------
        print('\n' + '-' * 72)
        print('阈值扫描  (先用原始 softmax 分数离线扫, 不重复检索)')
        print('-' * 72)
        raw = {}
        for q, expect, grp in queries:
            raw[(q, grp)] = softmax_best(bm25, q)
        for q in hard_neg:
            raw[(q, 'N硬负')] = softmax_best(bm25, q)
        for q in soft_neg:
            raw[(q, 'N软负')] = softmax_best(bm25, q)

        def hit_rate(items, t, grp, need_match=False):
            if not items:
                return '   -   '
            n = 0
            for q, e in items:
                s, m = raw[(q, grp)]
                if s >= t and (not need_match or m == e):
                    n += 1
            return '%5.1f%%' % (100.0 * n / len(items))

        a_items = [(q, e) for q, e, g in queries if g == 'A原题']
        b_items = [(q, e) for q, e, g in queries if g == 'B轻改写']
        hn_items = [(q, None) for q in hard_neg]
        sn_items = [(q, None) for q in soft_neg]

        head = '%-8s %-10s %-10s %-10s %-10s' % ('阈值', 'A拦截率', 'A正确率', 'B拦截率', 'B正确率')
        if hard_neg:
            head += '%12s %12s' % ('硬负假阳率', '软负拦截率')
        print(head)
        for t in THRESHOLDS:
            line = '%-8.2f %-10s %-10s %-10s %-10s' % (
                t, hit_rate(a_items, t, 'A原题'), hit_rate(a_items, t, 'A原题', True),
                hit_rate(b_items, t, 'B轻改写'), hit_rate(b_items, t, 'B轻改写', True))
            if hard_neg:
                line += '%12s %12s' % (hit_rate(hn_items, t, 'N硬负'), hit_rate(sn_items, t, 'N软负'))
            mark = '  <-- 当前默认' if abs(t - 0.85) < 1e-9 else ''
            print(line + mark)
        print('  注: 硬负假阳率越低越好(应为 0); 软负是"领域内开放问题", 拦截率仅作参考')

        # ---------------- 分层拦截 + 延迟 ----------------
        print('\n' + '-' * 72)
        print('分层拦截与延迟  (走真实 bm25.search 调用链, 阈值 0.85)')
        print('-' * 72)
        tiers = Counter()
        lat = {'bm25': [], 'redis': []}
        wrong = []

        for q, expect, grp in queries:
            if not args.keep_cache:
                redis_client.delete_data(q)
            cached_before = redis_client.get_data(q) is not None

            t0 = time.perf_counter()
            answer, need_rag = bm25.search(q, threshold=0.85)
            dt = (time.perf_counter() - t0) * 1000.0

            if answer:
                tier = 'redis' if cached_before else 'bm25'
                tiers['%s(%s)' % (tier, grp)] += 1
                lat[tier].append(dt)
                score, matched = raw[(q, grp)]
                if matched != expect:
                    wrong.append((q, expect, matched, score))
            else:
                tiers['rag(%s)' % grp] += 1

        for grp in ('A原题', 'B轻改写'):
            total = sum(1 for q, e, g in queries if g == grp)
            if not total:
                continue
            print('\n  【%s】共 %d 条' % (grp, total))
            for name in ('bm25', 'redis', 'rag'):
                n = tiers.get('%s(%s)' % (name, grp), 0)
                bar = '#' * int(round(40.0 * n / total))
                print('    %-6s %5d  %5.1f%%  %s' % (name, n, 100.0 * n / total, bar))

        print('\n  延迟 (毫秒):')
        for name in ('bm25', 'redis'):
            v = lat[name]
            if v:
                print('    %-6s n=%4d  P50=%.2f  P95=%.2f  max=%.2f' % (
                    name, len(v), pct(v, 0.50), pct(v, 0.95), max(v)))
            else:
                print('    %-6s 无样本' % name)

        if wrong:
            print('\n  高危: 拦截了但匹配到【错误】FAQ 的前 8 条 (阈值 0.85)')
            for q, e, m, s in wrong[:8]:
                print('    score=%.3f  问: %s' % (s, q[:30]))
                print('              期望: %s' % e[:30])
                print('              实得: %s' % m[:30])

        missed = [(q, e, g, raw[(q, g)][0]) for q, e, g in queries if raw[(q, g)][0] < 0.85]
        if missed:
            print('\n  未拦截(softmax < 0.85, 会走 RAG) 共 %d 条:' % len(missed))
            for q, e, g, s in missed:
                print('    [%s] score=%.3f' % (g, s))
                print('        问    : %s' % q[:40])
                print('        期望FAQ: %s' % e[:40])

        # ---------------- 负样本明细 ----------------
        if hard_neg:
            print('\n' + '-' * 72)
            print('负样本明细  (假阳性 = 本应走 RAG 却被 FAQ 层拦截)')
            print('-' * 72)
            for label, items, grp in (('硬负 · 明显域外', NEG_DOMAIN_OUT, 'N硬负'),
                                      ('硬负 · 短泛化(健康域内)', NEG_SHORT_GENERIC, 'N硬负'),
                                      ('软负 · 评测集(领域内开放)', soft_neg, 'N软负')):
                if not items:
                    continue
                fp = sum(1 for q in items if raw[(q, grp)][0] >= 0.85)
                print('\n  【%s】%d 条, 阈值 0.85 下假阳性 %d 条 (%.1f%%)'
                      % (label, len(items), fp, 100.0 * fp / len(items)))
                for q in items:
                    s, m = raw[(q, grp)]
                    flag = '   <== 假阳性' if s >= 0.85 else ''
                    print('    %.3f  %s%s' % (s, q[:30], flag))
                    if s >= 0.85:
                        print('           被误答为: %s' % m[:44])
            worst = max((raw[(q, 'N硬负')][0] for q in hard_neg), default=0.0)
            print('\n  硬负样本最高分 = %.3f  -> 阈值只要高于该值就不会误拦' % worst)

        # ---------------- 结论 ----------------
        print('\n' + '=' * 72)
        for grp in ('A原题', 'B轻改写'):
            total = sum(1 for q, e, g in queries if g == grp)
            if not total:
                continue
            fast = tiers.get('bm25(%s)' % grp, 0) + tiers.get('redis(%s)' % grp, 0)
            ok = sum(1 for q, e, g in queries
                     if g == grp and raw[(q, g)][0] >= 0.85 and raw[(q, g)][1] == e)
            print('%s: 拦截率 %.1f%% (%d/%d) | 正确率 %.1f%% (%d/%d)'
                  % (grp, 100.0 * fast / total, fast, total, 100.0 * ok / total, ok, total))
        print('=' * 72)
        print('说明: 拦截率 = 不再进入 RAG 的比例; 正确率 = 拦截且 Top-1 命中预期 FAQ 的比例')
        print()

    finally:
        mysql_client.close()


def main():
    ap = argparse.ArgumentParser(description='四级递进召回命中率基准')
    ap.add_argument('--sets', default='ABN', help='要测的查询组: A(原题)/B(轻改写)/N(负样本), 默认 ABN')
    ap.add_argument('--limit', type=int, default=None, help='只取前 N 条 FAQ(快速验证)')
    ap.add_argument('--keep-cache', action='store_true', help='不清理 Redis 缓存, 用于测热缓存层')
    ap.add_argument('--quiet', action='store_true', help='不打印检索层详细日志, 仅输出报告')
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
