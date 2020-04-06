#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:27:56 2020

@author: ansh
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer

torchvision_models_mean = [0.485, 0.456, 0.406]
torchvision_models_stDev = [0.229, 0.224, 0.225]

image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(torchvision_models_mean, torchvision_models_stDev)
        ]), 
    'val': transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize(torchvision_models_mean, torchvision_models_stDev),
        ]),
    'test': transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize(torchvision_models_mean, torchvision_models_stDev),
        ])
}



class gdDataset(Dataset):
    def __init__(self, df, image_path,
                 train_mode = True,
                 bert_model_name = 'bert-base-uncased',
                 transforms = None,
                 tokenizer = None):
        assert(not (train_mode==False and tokenizer==None)), \
            'tokenizer must be provided to gdDataset if train_mode == False'
        self.df = df
        self.bert_max_length = 25
        self.train_mode = train_mode
        self.image_path = image_path
        self.transforms = transforms
        self.bert_model_name = bert_model_name
        self.tokenizer = tokenizer if tokenizer else self._get_tfidf_vectorizer()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.num_classes = self.df.classes.nunique()
        # TODO : create a label_map dictionary
        
    def __len__(self):
        return self.df.__len__()
    
    def __getitem__(self, idx):
        
        text = self.df.loc[idx, 'text']
        # TODO : text = self.clean_text(text)
        tfidf_vector = self._get_tfidf_vector(text)
        bert_input_ids = self._get_bert_input_ids(text)

        # TODO : labels need to be converted to bert_input_ids as well
        label = self.df.loc[idx, 'classes'] if self.train_mode else None
        
        image_path = self.image_path + self.df.loc[idx, 'image']
        image = Image.open(image_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        return image, bert_input_ids, tfidf_vector, label

    def clean_text(self, text):
        # TODO : add text cleaning methods here
        pass
    
    def _get_label_bert_embeddings(self, label):
        # TODO
        pass
        
    def _get_bert_input_ids(self, text):
        # returns the input_ids for BertModel using BertTokenizer
        input_ids = self.bert_tokenizer.encode(text,
                                          return_tensors='pt',
                                          pad_to_max_length=True,
                                          max_length=self.bert_max_length)
        input_ids.squeeze_(0)
        return input_ids
    
    def _get_tfidf_vector(self, text):
        tfidf_vector = self.tokenizer.texts_to_matrix([text], mode='tfidf')
        tfidf_vector = torch.tensor(tfidf_vector, dtype=torch.float32).squeeze(0)
        return tfidf_vector
    
    def _get_tfidf_vectorizer(self):
        '''
        Should be used only when using trainind data.
        Fits a tokenizer object on the texts to create tfidf vectors.
        The vectorizer skips tokens that appear only once in the corpus.
            i.e. to be a valid term in the tf-idf matrix,
            a term must appear atleast twice across all documents.
            
        Returns
        -------
        tokenizer : keras.preprocessing.text.tokenizer
        '''
        corpus = list(self.df.text)
        # fit a tokenizer to get vocab size
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
        vocab_size = len([k for k,v in tokenizer.word_counts.items() if v > 1])
        # fit tokenizer with reduced vocabulary
        tokenizer = Tokenizer(num_words = vocab_size, oov_token = 'UNK')
        tokenizer.fit_on_texts(corpus)
        return tokenizer
