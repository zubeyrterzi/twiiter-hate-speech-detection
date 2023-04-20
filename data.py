import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re

train_file = 'train_E6oV3lV.csv'
test_file = 'test_tweets_anuFYb8.csv'

model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)

class TDataset_tokenizer(Dataset):
    def __init__(self, train= None):
        
        self.tokenizer = tokenizer
        if train == True:
            self.data = pd.read_csv(train_file)
            self. data.drop('id', axis= 1, inplace=True)
        elif train == False:
            self.data = pd.read_csv(test_file)
            self.data.drop('id', axis= 1, inplace=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tweet = self.data.iloc[index]['tweet']
        tweet = tweet.lower()
        tweet = re.sub(r'http\S+', '', tweet) # remove URLs
        tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet) # remove mentions
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet) # remove punctuation and numbers
        tweet = tweet.strip()
        input_ids = tokenizer(tweet).input_ids
        label = self.data.iloc[index]['label']
        return {"input_ids": input_ids, "label": label}
    

#dataset_tr = TDataset_tokenizer(train= True)
#tuple_ids = next(iter(dataset_tr))
#print('0', tuple_ids[0])
#print('1', tuple_ids[1])





    
        
 