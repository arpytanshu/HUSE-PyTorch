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


df = pd.read_csv('./resources/validation_data.csv').sample(12)
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
BertEmbeddingModel, bert_emb_dim = model.BertEmbeddingModel(BERT_MODEL_NAME)




####
# CONFIGURE dimensions in HUSE_config object

HUSE_config = model.HUSE_config()

HUSE_config.image_embed_dim = image_emb_dim
HUSE_config.bert_hidden_dim = bert_emb_dim
HUSE_config.tfidf_dim       = dataset.tokenizer.num_words
HUSE_config.num_classes     = dataset.num_classes


####
# Create HUSE model

HUSE_model = model.HUSE(ImageEmbeddingModel, BertEmbeddingModel, HUSE_config)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print('Device Type: ', device.type)

if device.type == 'cuda':
    HUSE_model.cuda()
    


for batch in dataloader:

    image = batch[0]
    bert_input_ids = batch[1]
    tfidf_vector = batch[2] 
    label = batch[3]

    huse_out = HUSE_model(image, bert_input_ids, tfidf_vector)
    
    print('\t\t *** new batch *** \t\t')
    print('image',  image.shape)
    print('bert_input_ids', bert_input_ids.shape)
    print('tfidf', tfidf_vector.shape)
    print(label.__len__())
    print(huse_out.shape)

