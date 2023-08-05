import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss
from model.model import SentimentClassifier
from loader.dataset import SentimentDataset
from losses.loss import FocalLoss

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging
from sklearn.model_selection import train_test_split
import warnings
import argparse
warnings.filterwarnings("ignore")
logging.set_verbosity_error()



def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/data_segment.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=86)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_split', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    opt = parser.parse_args()
    return opt

def prepare_loaders(df, batch_size, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    X_val = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = SentimentDataset(df_train, tokenizer, max_len=120)
    valid_dataset = SentimentDataset(X_val, tokenizer, max_len=120)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader

def train(model,criterion,optimizer, train_loader):
    model.train()
    losses = []
    correct = 0

    for data in tqdm(train_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)
        _, pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f'Train Accuracy: {correct.double()/len(train_loader.dataset)} Loss: {np.mean(losses)}')

def eval(model,criterion,valid_loader):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)
            correct += torch.sum(pred == targets)
            losses.append(loss.item())
    
    
    print(f'Valid Accuracy: {correct.double()/len(valid_loader.dataset)} Loss: {np.mean(losses)}')
    return correct.double()/len(valid_loader.dataset)

if __name__ == '__main__':
    args = parser_opt()
    seed_everything(args.seed)
    df = pd.read_csv(args.data_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(n_classes=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)
    X_train,X_test, y_train, y_test = train_test_split(df, df['label'],test_size = 0.1, random_state= 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.1, random_state= 42)
    train_df = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    
    #split fold
    skf = StratifiedKFold(n_splits=args.n_split)
    for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.label)):
        train_df.loc[val_, "kfold"] = fold
    
    for fold in range(skf.n_splits):
        print(f'-----------Fold: {fold+1} ------------------')
        train_loader, valid_loader = prepare_loaders(train_df,batch_size=args.batch_size, fold=fold)
        #model = SentimentClassifier(n_classes=2).to(device)
        criterion = FocalLoss(gamma=2, alpha=0.25)
        # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
        # Batchsize: 16, 32
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=0, 
                    num_training_steps=len(train_loader)*args.epochs
                )
        best_acc = 0
        for epoch in range(args.epochs):
            print(f'Epoch {epoch+1}/{args.epochs}')
            print('-'*30)

            train(model,criterion,optimizer, train_loader)
            val_acc = eval(model=model, criterion= criterion, valid_loader= valid_loader)

            if not os.path.exist('ckpt'):
                os.mkdir('ckpt')
            if val_acc > best_acc:
                torch.save(model.state_dict(), f'./ckpt/phobert_fold{fold+1}.pth')
                best_acc = val_acc

