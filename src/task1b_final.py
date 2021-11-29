#!/usr/bin/env python
# coding: utf-8

#!pip install transformers


#!pip install sentence-transformers


from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import xml.etree.ElementTree as ET
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import random
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-multi_lb_classi", default=1, type=int, help="whether train a multi-lable classifier or individual classifiers for each class")
opt = parser.parse_args()

files = os.listdir("./From-ScisummNet-2019")

test_docs  = os.listdir("./Test")

# train_end = int(len(files)*0.85)
# val_end = int(len(files)*0.15)+train_end
# train_docs = files[0:train_end]
# val_docs = files[train_end:val_end]
train_docs = files

def preprocess(example_sent):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(example_sent.lower())

    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = [w for w in filtered_sentence if w.isalpha()]
    new = " " 
    a = new.join(filtered_sentence)
    return a

    
def get_dataset(files,folder):
    data_lis = []
    label_lis = []
    for z,f in enumerate(files):
        print("f",f)
        try:
            citants = pd.read_json(folder+"/"+str(f)+"/citing_sentences.json")
            citants = citants[['citance_No','clean_text']]
            # queries = list(citants['clean_text'])
            # cite_no = list(citants.citance_No)
            tree = ET.parse(folder + "/"+f+"/Reference_XML/"+f+".xml")
            root = tree.getroot()
            final1=[]
            final2=[]
            i = 0
            total = len(root)
            for a in root:
                for b in a:
                    final1.append(b.text)
                    if i == 0:
                        final2.append("Abstract")
                    if i == 1:
                        final2.append("Introduction")
                    elif i < total-2:
                        final2.append("Experiment/Discussion")
                    if i == total-2 or i == total-1:
                        final2.append("Results/Conclusion")
                    if i == total:
                        final2.append("Acknowledgment")
                i = i+1

            d={'col1':final1,'col2':final2}

            rp = pd.DataFrame(data=d)
            data_lis.extend(list(rp.col1))
            label_lis.extend(list(rp.col2))
            
        except Exception as e: print(e)
    return data_lis, label_lis
            
    
train_data, train_labels = get_dataset(train_docs,folder='From-ScisummNet-2019')
# val_data,val_labels = get_dataset(val_docs,folder='From-ScisummNet-2019')
test_data,test_labels = get_dataset(test_docs,folder='Test')


le = preprocessing.LabelEncoder()
le.fit(train_labels)
train_labels = le.transform(train_labels)
# val_labels = le.transform(val_labels)
test_labels = le.transform(test_labels)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_data,return_tensors="pt",max_length=512,padding='max_length',truncation=True)
# val_encodings = tokenizer(val_data,return_tensors="pt",max_length=512,padding='max_length',truncation=True)
test_encodings = tokenizer(test_data,return_tensors="pt",max_length=512,padding='max_length',truncation=True)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, torch.from_numpy(train_labels))
# val_dataset = IMDbDataset(val_encodings, torch.from_numpy(val_labels))
test_dataset = IMDbDataset(test_encodings, torch.from_numpy(test_labels))


import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"

def calc_metrics(pred_lis, gt_lis):
    acc = accuracy_score(pred_lis, gt_lis)
    recall = recall_score(pred_lis, gt_lis,average='micro')
    prec = precision_score(pred_lis, gt_lis,average='micro')
    f1 = f1_score(pred_lis, gt_lis,average='micro')
    print("metrics obtained in test: accuracy {} recall {}, precision {}, f1-score {}".format(acc,recall,prec,f1))
    
def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({le.transform([cl])[0]:cl})

    return res


mapping = get_integer_mapping(le)
mapping

multi_lb_classi = True
if multi_lb_classi:

    training_args = TrainingArguments(
        output_dir='/ssd_scratch/cvit/results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='/ssd_scratch/cvit/logs',            # directory for storing logs
        logging_steps=1500,
    )

    # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model = nn.DataParallel(model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset          # training dataset
        # eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    model.eval()
    # trainer.evaluate()
    
    torch.save(model,"dummytask1b_bert.pth")
    # val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pred_lis = []
    gt_lis = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        pred_lis.extend(predictions.cpu().numpy())
        gt_lis.extend(batch["labels"].cpu().numpy())
        

    calc_metrics(pred_lis, gt_lis)


    gt_lis = np.array(gt_lis)
    pred_lis = np.array(pred_lis)


    #class-wise metrics
    for cl in mapping:
        print("for class: ",mapping[cl])
        ind = np.where(gt_lis==cl)
        small_pred = pred_lis[ind]
    #     print(small_pred, gt_lis[ind])
        calc_metrics(small_pred, gt_lis[ind])




# ## Classifiers for each class
else:
    training_args = TrainingArguments(
        output_dir='./ssd_scratch/cvit/results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='/ssd_scratch/cvit/logs',            # directory for storing logs
        logging_steps=1500,
    )


    for cl in mapping:
        print("for class: ",mapping[cl])
        train_ind = np.where(train_labels==cl)[0].astype(int)
        # val_ind = np.where(val_labels==cl)[0].astype(int)
        test_ind = np.where(test_labels==cl)[0].astype(int)
        
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        
        train_encodings = tokenizer(list(train_data[train_ind]),return_tensors="pt",max_length=512,padding='max_length',truncation=True)
        test_encodings = tokenizer(list(test_data[test_ind]),return_tensors="pt",max_length=512,padding='max_length',truncation=True)
        train_dataset = IMDbDataset(train_encodings, torch.from_numpy(train_labels[train_ind]))
        test_dataset = IMDbDataset(test_encodings, torch.from_numpy(test_labels[test_ind]))    
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset         # training dataset
        )


        trainer.train()
        torch.save(model, "dummybert_single_lb_classi"+str(cl)+".pth")
        # eval
        val_dataloader = DataLoader(test_dataset, batch_size=8)
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pred_lis = []
        gt_lis = []

        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            pred_lis.extend(predictions.cpu().numpy())
            gt_lis.extend(batch["labels"].cpu().numpy())
        
        calc_metrics(pred_lis, gt_lis)



