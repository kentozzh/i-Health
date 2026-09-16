import warnings
warnings.filterwarnings("ignore")


import os, json, torch, numpy as np

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(module_path, 'models')

from base import logger
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class IntentRecognizer:
    def __init__(self, load_model_path = os.path.join(model_path, 'bert_intent_recognizer')):
        self.load_model_path = load_model_path
        self.logger = logger
        self.bert_path = os.path.join(module_path, 'models', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'使用设备: {self.device}')
        self.label_dict = {'通用问题':0, '专业问题':1}

        self.__load_model()

    def __load_model(self):
        try:
            if os.path.exists(self.load_model_path):
                self.model = BertForSequenceClassification.from_pretrained(self.load_model_path)
                self.model.to(self.device)
                self.logger.info(f'模型已加载: {self.load_model_path}')
            else:
                self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels=2)
                self.model.to(self.device)
                self.logger.info('初始化Bert模型完成')

                self.__train_model()
        except Exception as e:
            self.logger.error(f'加载模型失败: {e}')

    def __save_model(self):
        self.model.to(self.device)
        self.model.save_pretrained(self.load_model_path)
        self.tokenizer.save_pretrained(self.load_model_path)
        self.logger.info(f'模型已保存: {self.load_model_path}')

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'         # 返回张量
        )
        return encodings, [self.label_dict[label] for label in labels]

    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        return Dataset(encodings, labels)

    def __train_model(self, train_data_path = os.path.join(module_path, 'bert_train_data', 'bert专业通用问题分类500条.json')):
        try:
            with open(train_data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(value) for value in f.readlines()]
        except FileNotFoundError as e:
            self.logger.error(f'文件未找到: {e}')
            raise

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        train_x, test_x, train_y, test_y = train_test_split(texts, labels, train_size=0.8, random_state=88)
        train_encoding, train_label = self.preprocess_data(train_x, train_y)
        test_encoding, test_label = self.preprocess_data(test_x, test_y)

        train_args = TrainingArguments(
            output_dir=model_path,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=20,
            weight_decay=0.01,
            logging_dir=os.path.join(model_path, 'bert_logs'),
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model='eval_loss',
            fp16=False
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.create_dataset(train_encoding, train_label),
            eval_dataset=self.create_dataset(test_encoding, test_label),
            compute_metrics=self.compute_metrics
        )

        try:
            self.logger.info('开始训练模型')
            trainer.train()
            self.__save_model()
        except Exception as e:
            self.logger.error(f'训练模型失败: {e}')


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        accuracy = (pred == labels).mean()
        return {"accuracy":accuracy}

    def evaluate_model(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        dataset = self.create_dataset(encodings, labels)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)

        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用问题", "专业问题"]
        ))

        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))


    def predict_category(self, query):
        if self.model is None:
            logger.error("模型未训练或加载!")
            return '通用问题'

        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return '专业问题' if prediction == 1 else '通用问题'


if __name__ == '__main__':
    intent_recognizer = IntentRecognizer()
    # intent_recognizer.train_model()
    print(intent_recognizer.predict_category('如何保持身心健康？'))
    print(intent_recognizer.predict_category('月亮为什么会发光？'))