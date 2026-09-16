# =====================================================================
# 意图分类模型评测 (eval_intent.py)
#
#   回答"500 条语料, 验证集准确率多少?"这个问题。
#
#   复现训练时的同一划分(train_size=0.8, random_state=88), 在验证集上评估
#   已落盘的 rag/models/bert_intent_recognizer。
#
#   用法: python eval_intent.py
# =====================================================================

import importlib.util
import json
import os
import sys

import numpy as np

from base import Config

# 直接从文件加载模块, 避免 import rag 触发 vector_store 的模型加载
_CORE = os.path.join(Config().PROJECT_ROOT, 'rag', 'core')
_spec = importlib.util.spec_from_file_location(
    'intent_recognizer_mod', os.path.join(_CORE, 'intent_recognizer.py'))
_mod = importlib.util.module_from_spec(_spec)
sys.modules['intent_recognizer_mod'] = _mod
_spec.loader.exec_module(_mod)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

TRAIN_DATA = 'rag/bert_train_data/bert专业通用问题分类500条.json'


def _eval_args():
    """
    显式给出 output_dir。

    Trainer 在不传 args 时会把 output_dir 默认设为【当前工作目录】下的 "tmp_trainer"
    (见 transformers/trainer.py), 会在项目根目录留下一个空目录 -> 指到临时目录避免污染。
    """
    import tempfile
    return TrainingArguments(output_dir=tempfile.mkdtemp(prefix='intent_eval_'),
                             per_device_eval_batch_size=8,
                             report_to=[])


def main():
    path = os.path.join(Config().PROJECT_ROOT, TRAIN_DATA)
    with open(path, encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    texts = [item['query'] for item in data]
    labels = [item['label'] for item in data]

    # 与训练时完全相同的划分
    _, test_x, _, test_y = train_test_split(
        texts, labels, train_size=0.8, random_state=88)

    recognizer = _mod.IntentRecognizer()
    label_dict = recognizer.label_dict                      # {'通用问题':0, '专业问题':1}
    inv = {v: k for k, v in label_dict.items()}
    y_true = [label_dict[label] for label in test_y]

    encodings = recognizer.tokenizer(
        test_x, truncation=True, padding=True, max_length=128, return_tensors='pt')
    dataset = recognizer.create_dataset(encodings, y_true)

    trainer = Trainer(model=recognizer.model, args=_eval_args())
    output = trainer.predict(dataset)
    y_pred = np.argmax(output.predictions, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print()
    print('=' * 66)
    print('意图分类模型评测 (bert-base-chinese 微调, 通用/专业二分类)')
    print('=' * 66)
    print('训练语料      : %d 条' % len(data))
    print('划分方式      : train_size=0.8, random_state=88 (与训练一致)')
    print('验证集规模    : %d 条' % len(test_x))
    print('验证集分布    : %s' % {'通用问题': y_true.count(0), '专业问题': y_true.count(1)})
    print('模型路径      : rag/models/bert_intent_recognizer')
    print()
    print('验证集准确率  : %.4f' % acc)
    print()
    print('分类报告:')
    print(classification_report(y_true, y_pred,
                                target_names=['通用问题', '专业问题'], digits=4))
    print('混淆矩阵 (行=真实, 列=预测):')
    print('                预测通用  预测专业')
    print('  真实通用      %6d %8d' % (cm[0][0], cm[0][1]))
    print('  真实专业      %6d %8d' % (cm[1][0], cm[1][1]))
    print()
    err = [(test_x[i], inv[y_true[i]], inv[y_pred[i]])
           for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    if err:
        print('误判样本 (%d 条):' % len(err))
        for q, t, p in err[:10]:
            print('   %s' % q[:34])
            print('        真实: %s | 预测: %s' % (t, p))
    else:
        print('验证集上无误判样本。')
    print('=' * 66)
    print()


if __name__ == '__main__':
    main()
