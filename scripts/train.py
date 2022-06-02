import argparse
import logging
import os

import torch
import numpy as np
import pandas as pd
import re

from transformers import BertForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, ElectraForSequenceClassification

from sklearn.metrics import label_ranking_average_precision_score

try:
    from sagemaker_inference import environment
except:
    from sagemaker_training import environment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]
bucket = 'pytorch-unsmile-demo'

class Unsmile_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(x):
    return {
        'lrap': label_ranking_average_precision_score(x.label_ids, x.predictions),
    }

def make_label_list(row):
    temp_list = []
    for col in unsmile_labels:
        temp_list.append(row[col])
    return temp_list

def pre_processing(text):
    text = re.sub('[^ㄱ-힣a-zA-Z0-9 ]', ' ', text)    
    text = re.sub(' +', ' ', text)
    
    result_text = text[0]
    cnt = 0
    
    for alpha in text[1:]:
        if result_text[-1] == alpha: cnt += 1
        else: cnt = 0

        if cnt < 3: result_text += alpha
        else: continue
        
    return result_text

def train(args):

    # load dataset
    train_file = 'data/unsmile_train_v1.0.tsv'
    test_file = 'data/unsmile_valid_v1.0.tsv'

    s3_uri_train = 's3://{}/{}'.format(bucket, train_file)
    s3_uri_test = 's3://{}/{}'.format(bucket, test_file)
    
    train_df = pd.read_csv(s3_uri_train, sep='\t')
    test_df = pd.read_csv(s3_uri_test, sep='\t')
    
    # when run with preprocessing data
    train_df['문장2'] = train_df['문장'].apply(lambda x: pre_processing(x))
    test_df['문장2'] = test_df['문장'].apply(lambda x: pre_processing(x))
    
    # plain: 문장, preprocessing: 문장2
    train_sentence_list = list(train_df['문장2'])
    test_sentence_list = list(test_df['문장2'])

    # tokenize sentence & make datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_sentence_token = tokenizer(train_sentence_list)
    test_sentence_token = tokenizer(test_sentence_list)
    
    train_df['labels'] = train_df.apply(lambda x: make_label_list(x), axis=1)
    test_df['labels'] = test_df.apply(lambda x: make_label_list(x), axis=1)
    
    train_label_list = list(train_df['labels'])
    test_label_list = list(test_df['labels'])
    
    train_dataset = Unsmile_Dataset(train_sentence_token, train_label_list)
    test_dataset = Unsmile_Dataset(test_sentence_token, test_label_list)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_labels=len(unsmile_labels) # Label 갯수

    if 'electra' in args.model_name.lower():
        model = ElectraForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels, 
            problem_type="multi_label_classification"
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels, 
            problem_type="multi_label_classification"
        )

    model.config.id2label = {i: label for i, label in zip(range(num_labels), unsmile_labels)}
    model.config.label2id = {label: i for i, label in zip(range(num_labels), unsmile_labels)}
    
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='lrap',
        greater_is_better=True,
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset, 
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    print("======= Training Job Finished =======")
    
    trainer.save_model(args.model_dir)
    print("======= Complete Saving Model =======")
    print("======= End =======")
    

if __name__ == "__main__":
    
    print('======= train.py start!! ========')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ["SM_OUTPUT_DIR"]
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default='beomi/kcbert-base'
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=16
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/opt/ml/checkpoints'
    )
    parser.add_argument(
        "--temp_param",
        type=str,
        default='/opt/ml/checkpoints'
    )
    print('============================')
    print(parser.parse_args())
    print('============================')

    #train(parser.parse_args())