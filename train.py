import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
import data
import model as mdl
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F


print('MMMMX')
training_data = data.TDataset_tokenizer(train= True)
test_data = data.TDataset_tokenizer(train= False)

model_name = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_name)



model = mdl.TransformerForSequenceClassification(config)

#padding function to set all tensors same size
def collate_fn(batch):
    input_ids = [sample['input_ids'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    input_ids_padded = []
    for ids in input_ids:
        if len(ids) > 0:
            max_len = max(len(ids) for ids in input_ids)
            padded_ids = ids + [0] * (max_len - len(ids))
            input_ids_padded.append(padded_ids)
    if len(input_ids_padded) == 0:
        return None
    return {'input_ids': torch.tensor(input_ids_padded), 
            'label': torch.tensor(labels)}
train_dataloader = DataLoader(dataset= training_data, batch_size= 8, collate_fn= collate_fn)
test_dataloader = DataLoader(dataset= test_data, batch_size= 8, collate_fn= collate_fn)

optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.BCEWithLogitsLoss()

model.train()

for epoch in range(8):
    for batch in train_dataloader:
        if batch is None:
            continue
        input_ids = batch['input_ids']
        labels = batch['label']
        labels = F.one_hot(labels, num_classes=2).to(torch.float)
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    print(f"Epoch {epoch+1}")



