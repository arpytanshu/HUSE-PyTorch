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
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F



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
        self.classes_map = self._get_label_map()
        self.df['classes_map'] = df.classes.map(self.classes_map)
    
    def __getitem__(self, idx):
        
        text = self.df.loc[idx, 'text']
        # TODO : text = self.clean_text(text)
        tfidf_vector = self._get_tfidf_vector(text)
        bert_input_ids = self._get_bert_input_ids(text)
        label = self.df.loc[idx, 'classes_map'] if self.train_mode else None
        image_path = self.image_path + self.df.loc[idx, 'image']
        image = Image.open(image_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, bert_input_ids, tfidf_vector, label

    def clean_text(self, text):
        # TODO : add text cleaning methods here
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
        '''
        Uses Keras.preprocessing.text.Tokenizer object created by _get_tfidf_vectorizer
        to generate tfidf feature vectors for samples.

        Returns
        -------
        tfidf_vector : torch tensor

        '''
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
    
    def _get_label_map(self):
        '''
        Creates a mapping from class labels to integer ids.
        The mapping is used for creating the Semantic Graph later.
        Returns
        -------
        A dictionary mapping from class names to integers.
        '''
        _, class_names = self.df.classes.factorize(sort=True)
        mapping = {k:v for v,k in enumerate(class_names)}
        return mapping
    
    def __repr__(self):
        # TODO
        return 'len:{} classes:{}'.format(len(self.df), self.num_classes)

    def __len__(self):
        return self.df.__len__()



class BertSentenceEncoder():
    def __init__(self, model_name='bert-base-cased'):
        '''
        Uses a pre-trained bert to embed sentences and pool them along the tokens.
        
        Parameters
        ----------
        model_name : string, optional
            DESCRIPTION. The default is 'bert-base-cased'.
            
            Find a list of usable pre-trained bert models from:
                https://huggingface.co/transformers/pretrained_models.html
        '''

        self.model_name =   model_name
        self.config =       BertConfig.from_pretrained(self.model_name, output_hidden_states=True, training=False)
        self.model =        BertModel.from_pretrained(self.model_name, config=self.config)
        self.tokenizer =    BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model.requires_grad_(False)
        self.model = self.model.to(self.device)
        
    def __repr__(self):
        return 'BertSentenceEncoder model:{}'.format(self.model_name)
    
    def _mean_pooler(self, encoding):
        return encoding.mean(dim=1)
    
    def _max_pooler(self, encoding):
        return encoding.max(dim=1).values
    
    def encoder(self, sentences, layer=-2, max_length=20, pooler='mean' ):
     
        assert isinstance(sentences, list), \
            "parameter 'sentences' is supposed to be a list of string/s"
        assert all(isinstance(x, str) for x in sentences), \
            "parameter 'sentences' must contain strings only"
        
        '''
        model(input_tokens) returns a tuple of 3 elements.
        out[0] : last_hidden_state  of shape [ B x T x D ]
        out[1] : pooler_output      of shape [ B x D ]
        out[2] : hidden_states      13 tuples, one for each hidden layer
                                    each tuple of shape [ B x T x D ]        
        '''
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', max_length=max_length)['input_ids']
            input_ids = input_ids.to(self.device)
            encoded = self.model(input_ids)
        if pooler == 'max':
            pooling_fn = self._max_pooler
        else: # anythig else defaults to mean-pooling
            pooling_fn = self._mean_pooler
        pooled = pooling_fn(encoded[2][layer])
        return pooled



def _fix_class_name(sample):
    sample = sample.replace('-', ' ').split('<')
    return sample



def SemanticGraph(classes_map):
    '''
    Creates a Semantic Similarity Graph using cosine distance between two class name embeddings
    The Class Name Embeddings are created using Bert Pre-trained model.
    The embeddings from the second last layer is mean-pooled.
    
    Parameters
    ----------
    classes_map : dict
        dictionary mapping class names to their integer ids

    Returns
    -------
    Semantic Graph
    A torch tensor of dimension [ num_classes x num_classes x embedding_dim ]
    '''

    BE = BertSentenceEncoder('bert-base-uncased')
    num_classes = classes_map.__len__()
    embed_dim = 3 * BE.model.pooler.dense.in_features
    
    # A 0 tensor for the semantic graph { S_G }
    S_G = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    
    # to temporarily hold the class names embeddings { CNE }
    CNE = torch.zeros((num_classes, embed_dim), dtype = torch.float32)
    
    # get embeddings for all class names
    for class_name,index in classes_map.items():
        class_name = _fix_class_name(class_name)
        CNE[index] = BE.encoder(class_name, pooler='mean').reshape(-1) # expected dimension => [3 x 768] = [2304]
    
    # create the Semantic Graph using cosine distance between class name embeddings
    for iy in range(num_classes):
        for ix in range(iy, num_classes):
            cos_sim = F.cosine_similarity(CNE[iy].reshape(1,-1), CNE[ix].reshape(1,-1))
            S_G[ix][iy] = S_G[iy][ix] = cos_sim
    return S_G

