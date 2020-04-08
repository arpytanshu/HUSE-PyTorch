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

df = pd.read_csv('./resources/validation_data.csv').sample(16)
df.reset_index(drop=True, inplace=True)

IMAGE_PATH = './resources/val_images/'
BATCH_SIZE = 4
BERT_MODEL_NAME = 'bert-base-uncased'

# Dataset & DataLoader 
dataset = utils.gdDataset(df.copy(),
                          image_path = IMAGE_PATH,
                          train_mode = True,
                          bert_model_name = BERT_MODEL_NAME,
                          transforms = utils.image_transforms['train'],
                          tokenizer = None # also provide tokenizer if this is a test dataset
                          )

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# create Semantic Graph
A = utils.SemanticGraph(dataset.classes_map)


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
# Create HUSE model & Losses

HUSE_model = model.HUSE(HUSE_config)

EmbeddingLoss = model.EmbeddingSpaceLoss(A, HUSE_config) 




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print('Device Type: ', device.type)

if device.type == 'cuda':
    HUSE_model.cuda()
    


for batch in dataloader:

    image = batch[0]
    bert_input_ids = batch[1]
    tfidf_vector = batch[2] 
    labels = batch[3].to(torch.int64)

    with torch.no_grad():
        image_embedding = ImageEmbeddingModel(image)
        bert_embedding = model.bert_embeddings_pooler(BertEmbeddingModel(bert_input_ids), HUSE_config.num_bert_layers)
        text_embedding = torch.cat([bert_embedding, tfidf_vector], dim=1)
    
    universal_embedding = HUSE_model(image_embedding, text_embedding)
    
    '''
    For CrossModalLoss, we get the universal embeddings from the HUSE_model
    using only either image or text at once.
    
    For obtaining universal embedding for text, we usa a Zero matrix for representing the image.
    For obtaining universal embedding for image, we usa a Zero matrix for representing the text.
    '''
    UnivEmb_only_image = HUSE_model(image_embedding, torch.zeros(text_embedding.shape))
    UnivEmb_only_text  = HUSE_model(torch.zeros(image_embedding.shape), text_embedding)
    
    # loss1 = model.ClassificationLoss()
    # loss2 = model.CrossModalLoss()
    # loss3 = model.SemanticSimilarityLoss(HUSE_config.margin, A)    

    
    # _ = loss1(universal_embedding, labels)
    # _ = loss2(UnivEmb_only_image, UnivEmb_only_text)
    # _ = loss3(universal_embedding, labels)
    
    loss = EmbeddingLoss(universal_embedding, UnivEmb_only_image, UnivEmb_only_text, labels)
    
    
    
    
    
    
    print('\t\t *** new batch *** \t\t')
    print('image_embedding',  image_embedding.shape)
    print('bert_embedding',  bert_embedding.shape)
    print('text_embedding', text_embedding.shape)
    
    print('universal_embedding', universal_embedding.shape)
    print('universal_embedding', universal_embedding.shape)
    print('loss', loss.shape)
    
