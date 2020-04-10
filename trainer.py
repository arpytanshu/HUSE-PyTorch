#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 02:19:41 2020

@author: ansh
"""


import utils
import model
import torch
import time
import pandas as pd
from torch.optim import RMSprop
from torch.utils.data import DataLoader


#%%
# ======== === ====== #
# NOTEBOOK RUN CONFIG #
# ======== === ====== #

DEV_MODE = True  # for dvelopment runs only

IMAGE_PATH = './resources/val_images/'
DATAFRAME_PATH = './resources/validation_data.csv'

BERT_MODEL_NAME = 'bert-base-uncased'
OPTIM_LEARNING_RATE = 1.6192e-05
OPTIM_MOMENTUM = 0.9
NUM_EPOCHS = 2 if DEV_MODE else 50
BATCH_SIZE = 4 if DEV_MODE else 1024

NUM_DEV_SAMPLES = 50 # number of samples to use in DEV_MODE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


#%%


# get dataframes, dataset and dataloader
# --- ----------- ------- --- ----------
tr_df = pd.read_csv(DATAFRAME_PATH)
if DEV_MODE:
    tr_df = tr_df.sample(NUM_DEV_SAMPLES)
tr_df.reset_index(drop=True, inplace=True)
tr_dataset = utils.gdDataset(tr_df.copy(),
                          image_path = IMAGE_PATH,
                          train_mode = True,
                          bert_model_name = BERT_MODEL_NAME,
                          transforms = utils.image_transforms['train'],
                          tokenizer = None) # also provide tokenizer if this is a test dataset
tr_dataloader = DataLoader(tr_dataset, batch_size = BATCH_SIZE, shuffle = True)


# create Semantic Graph
# ------ -------- -----
A = utils.SemanticGraph(tr_dataset.classes_map)

# Create image and text feature extraction models
# ------ ----- --- ---- ------- ---------- ------
ImageEmbeddingModel, image_emb_dim = model.ImageEmbeddingModel('resnet18')
BertEmbeddingModel, bert_emb_dim = model.BertEmbeddingModel(BERT_MODEL_NAME)

# CONFIGURE dimensions in HUSE_config object
# --------- ---------- -- ----------- ------
HUSE_config = model.HUSE_config()
HUSE_config.image_embed_dim = image_emb_dim
HUSE_config.bert_hidden_dim = bert_emb_dim
HUSE_config.tfidf_dim       = tr_dataset.tokenizer.num_words
HUSE_config.num_classes     = tr_dataset.num_classes

# Create HUSE model, Loss, Optimizer & set PyTorch device
# ------ ---- ------ ---- - -------- - --- ------- ------
HUSE_model = model.HUSE(HUSE_config)
EmbeddingLoss = model.EmbeddingSpaceLoss(A, device, HUSE_config) 
optimizer = RMSprop(HUSE_model.parameters(), lr=OPTIM_LEARNING_RATE, momentum=OPTIM_MOMENTUM)
if device.type == 'cuda':
    HUSE_model.cuda()
    BertEmbeddingModel.cuda()
    ImageEmbeddingModel.cuda()
   

#%%


def train_model(HUSE_model, dataloader, criterion, optimizer):    
    since = time.time()
    train_loss_hist = []
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
        print('-'*10)
        running_loss = 0.0
        for batch_ix, batch in enumerate(dataloader):
            batch_time = time.time()
            # set device for training data
            image = batch[0].to(device)
            bert_input_ids = batch[1].to(device)
            tfidf_vector = batch[2].to(device) 
            label = batch[3].to(torch.int64).to(device)
            # get image & text embeddings 
            with torch.no_grad():
                image_embedding = ImageEmbeddingModel(image)
                bert_embedding = model.bert_embeddings_pooler(BertEmbeddingModel(bert_input_ids), HUSE_config.num_bert_layers)
                text_embedding = torch.cat([bert_embedding, tfidf_vector], dim=1)

            optimizer.zero_grad()
            HUSE_model.zero_grad()

            with torch.set_grad_enabled(True):
                universal_embedding = HUSE_model(image_embedding, text_embedding)
                UnivEmb_only_image = HUSE_model(image_embedding, torch.zeros(text_embedding.shape, device=device))
                UnivEmb_only_text  = HUSE_model(torch.zeros(image_embedding.shape, device=device), text_embedding)
                
                loss = criterion(universal_embedding, UnivEmb_only_image, UnivEmb_only_text, label)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * image.size(0)
            batch_time_elapsed = time.time() - batch_time
            if((batch_ix+1)%10 == 0):
                print('epoch: {} batch: {}/{} batch_loss: {} time_in_batch: {}'.format(epoch+1, batch_ix+1, len(tr_dataloader), loss.item(), batch_time_elapsed))

        epoch_loss = running_loss / len(dataloader.dataset)
        print('Loss: {}'.format(epoch_loss))
        train_loss_hist.append(epoch_loss)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return HUSE_model, train_loss_hist

#%%


train_model(HUSE_model, tr_dataloader, EmbeddingLoss, optimizer)