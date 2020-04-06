#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:27:56 2020

@author: ansh
"""

import utils
import model
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer


df = pd.read_csv('./resources/validation_data.csv')
df.reset_index(drop=True, inplace=True)

IMAGE_PATH = './resources/val_images/'
BATCH_SIZE = 4
BERT_MODEL_NAME = 'bert-base-uncased'

# Dataset & DataLoader 
dataset = utils.gdDataset(df,
                          image_path = IMAGE_PATH,
                          train_mode = True,
                          bert_model_name = BERT_MODEL_NAME,
                          transforms = utils.image_transforms['train'],
                          tokenizer = None # also provide tokenizer if this is a test dataset
                          )

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)



# Create image and text feature extraction models
ImageEmbeddingModel, image_emb_dim = model.ImageEmbeddingModel('resnet18')
BertEmbeddingModel = model.BertEmbeddingModel(BERT_MODEL_NAME)




####
# CONFIGURE dimensions in HUSE_config object

HUSE_config = model.HUSE_config()












device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print('Device Type: ', device.type)


if device.type == 'cuda':
    image_emb_model.cuda()
    bert_emb_model.cuda()


for batch in dataloader:
    image = batch[0]
    text = batch[1]
    label = batch[2]
    
    tokenized_text = tokenizer.batch_encode_plus(text, return_tensors='pt', pad_to_max_length=True)['input_ids']
    tokenized_text = tokenized_text.to(device)
    image = image.to(device)
    
    
    img_emb = image_emb_model(image)
    _bert_out = bert_emb_model(tokenized_text)
    _hidden_states = _bert_out [2]
    
    _bert_emb = model.bert_embeddings_pooler(_hidden_states)
    _random_tfidf = torch.rand(batch_size, HUSE_config.tfidf_dim)
    text_emb =  torch.cat([_bert_emb, _random_tfidf], dim=1)
    
    
    image_tower_out = image_tower_model(img_emb)
    text_tower_out = text_tower_model(text_emb)
    
    shared_layer_out = None
    
    print('\t\t *** new batch *** \t\t')
    print(img_emb.shape)
    print(text_emb.shape)
    print()
    print(image_tower_out.shape)
    print(text_tower_out.shape)
    












from keras.pre

corpus = ['a b c','b g q','c b d a','d x t e','e c']
