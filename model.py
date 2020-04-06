#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:27:56 2020

@author: ansh
"""

import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models
from transformers import BertModel, BertConfig


'''
Methods / Classes for getting Image Embeddings
We use pre-trained models trained on ImageNet from torchvision
to get 1000-Dimentional vector representation for images.
'''


pretrained_imagenet_models = {
    'resnet18'      : models.resnet18,
    'resnet101'     : models.resnet101,
    'resnet152'     : models.resnet152,
    'resnext101'    : models.resnext101_32x8d,
    'resnext50'     : models.resnext50_32x4d,
    'default'       : models.resnet18
}

def ImageEmbeddingModel(model_name):
    '''
    Returns a PyTorch pre-trained model.
    Valid model names from torchvision can be added to pretrained_models dictionary.
    
    Parameters
    ----------
    
    model_name : string
        A valid model name having an entry in the pretrained_models dictionary.
        If the model_name provided is not valid, a default is used.
        
    features_extract : bool, optional
         If True, sets requires_grad attribute of model's parameter to False
                    
    '''

    if model_name not in pretrained_imagenet_models.keys():
        model_name = 'default'
        print('Invalid model name for ImageEmbeddingModel. Using default {}'.\
              format(pretrained_imagenet_models.get('default')))
        
    model = pretrained_imagenet_models.get(model_name)(pretrained=True)
    
    # freeze model parameters
    model.requires_grad_(False)

    out_dimension = model.fc.out_features

    return model, out_dimension







'''
Methods / Classes for getting BERT Embeddings text.
'''

'''
NOTE:
The HUSE paper described the text description for each sample (text modal)
to be much greater than 512 tokens for the dataset it used { UPMC Food-101 } 
and thus required to extract 512 most important tokens from the text description.

However the textual description of the samples in the dataset shared with us is 
much smaller than 512, thus eliminating the need for such token clipping.
'''



pretrained_bert_models = [  'bert-base-cased',
                            'bert-large-cased',
                            'bert-base-uncased',
                            'bert-large-uncased',
                            ]

def BertEmbeddingModel(model_name):
    '''
    Returns a PyTorch pre-trained model.
    
    Parameters
    ----------
    model_name : string
        A valid bert model name from huggingface's transformers.
        see: https://huggingface.co/transformers/pretrained_models.html

    Returns
    -------
    bert_model : A transformers Pre trained Bert model
        
        usage:
        ------    
        out = bert_model(input_tokens)
        
        input_tokens is a pytorch tensor of tokenized sentences that are to be embedded.
            see : BertTokenizer
            
        out is a tuple of 3 elements.
            out[0] : last_hidden_state  of shape [ B x T x D ]
            out[1] : pooler_output      of shape [ B x D ]
            out[2] : hidden_states      13 tuples, one for each hidden layer
                                    each tuple of shape [ B x T x D ] .
                                    
                                    B : batch_size
                                    T : sequence_length / time_dimension
                                    D : hidden_size [ 768 for base model, 1024 for large model ]
                                    
                                    out[2][-1] : embeddings from last layer
                                    out[2][1] : embeddings from first layer
    '''
    
    if model_name not in pretrained_bert_models:
        print('Invalid model name for BertEmbeddingModel. Using default bert-base-cased.')
        model_name = 'bert-base-cased'
    bert_config =       BertConfig.from_pretrained(model_name, output_hidden_states=True, training=False)
    bert_model =        BertModel.from_pretrained(model_name, config=bert_config)
    
    # freeze parameters
    bert_model.requires_grad_(False)
    
    out_dimension = bert_config.hidden_size
    
    return bert_model, out_dimension




'''
HUSE model
'''


class HUSE_config():
    def __init__(self):
        self.tfidf_dim = None
        self.num_classes = None 
        self.image_embed_dim = None
        self.bert_hidden_dim = None
        self.num_bert_layers = 4 # as in HUSE paper
        self.image_tower_hidden_dim = 512 # as in HUSE paper
        self.text_tower_hidden_dim = 512 # as in HUSE paper
    
    def __repr__(self):
        return ('Configuration object for HUSE model.\n'
            'image_embed_dim:\t\t\t{}\n'
            'bert_hidden_dim:\t\t\t{}\n'
            'num_bert_layers:\t\t\t{}\n'
            'image_tower_hidden_dim:\t{}\n'
            'text_tower_hidden_dim:\t{}\n'
            'tfidf_dim:\t\t\t\t{}\n'
            'num_classes:\t\t\t\t{}\n').format(
            self.image_embed_dim, self.bert_hidden_dim,
            self.num_bert_layers, self.image_tower_hidden_dim,
            self.text_tower_hidden_dim, self.tfidf_dim,
            self.num_classes)

    

'''
Image Tower
'''
class ImageTower(nn.Module):
    def __init__(self, config : HUSE_config):
        
        super(ImageTower, self).__init__()
        self.drop_prob = 0.15
        self.input_size = config.image_embed_dim
        self.hidden_size  = config.image_tower_hidden_dim
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, X):
        out = F.dropout( F.relu( self.fc1( X ) ), p = self.drop_prob, training = self.training)
        out = F.dropout( F.relu( self.fc2(out) ), p = self.drop_prob, training = self.training)
        out = F.dropout( F.relu( self.fc3(out) ), p = self.drop_prob, training = self.training)
        out = F.dropout( F.relu( self.fc4(out) ), p = self.drop_prob, training = self.training)
        out = F.relu( self.fc5(out) )
        out = F.normalize(out, p=2, dim=1) # L2 - Normalize
        return out


'''
Text Tower
'''
class TextTower(nn.Module):
    def __init__(self, config : HUSE_config):
        super(TextTower, self).__init__()
        
        self.drop_prob = 0.15
        self.input_size = config.tfidf_dim + config.bert_hidden_dim*config.num_bert_layers
        self.hidden_size  = config.text_tower_hidden_dim
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, X):
        out = F.relu(self.fc1(X))
        out = F.dropout(out, p = self.drop_prob, training = self.training)
        out = F.relu( self.fc2(out) )
        out = F.normalize(out, p=2, dim=1) # L2 - Normalize
        return out





class HUSE(nn.Module):
    def __init__(self, ImageEmbeddingModel, BertEmbeddingModel, config):
        super(HUSE, self).__init__()
        self.config = config
        self.ImageEmbeddingModel = ImageEmbeddingModel
        self.BertEmbeddingModel = BertEmbeddingModel
        self.ImageTower = ImageTower(config)
        self.TextTower = TextTower(config)
        self.shared_fc_layer = nn.Linear(
            in_features = config.image_tower_hidden_dim + config.text_tower_hidden_dim,
            out_features = config.num_classes)
        

    def forward(self, image, bert_input_ids, text_tfidf):
        
        with torch.no_grad():
            _bert_out = self.BertEmbeddingModel(bert_input_ids)
            _bert_embedding = self._bert_embeddings_pooler(_bert_out)
            # return _bert_embedding
            text_embedding = torch.cat([_bert_embedding, text_tfidf], dim=1)
            image_embedding = self.ImageEmbeddingModel(image)    

        out1 = self.ImageTower(image_embedding)
        out2 = self.TextTower(text_embedding)
        out = self.shared_fc_layer(torch.cat([out1, out2], dim=1))

        return out
    
    def _mean_pooler(self, encoding):
        return encoding.mean(dim=1)
    
    def _bert_embeddings_pooler(self, bert_out):
        '''
        Concatenates embeddings from hidden layers whose index are provided in layers_to_use.
        The embedding for each layer is mean pooled along the time / sequence dimension.
        Example
        -------
        This concatenate the embeddings from the last config.num_bert_layers layers for each token 
        and then average all token embeddings for a sequence.
            out = BertModel(sinput_ids)
            embeddings = bert_embeddings_pooler(out)
        '''
        hidden_states = bert_out[2]
        layers_to_use = range(self.config.num_bert_layers, 0, -1)
        
        pooled = None
        for layer in layers_to_use:
            if pooled is None:
                pooled = self._mean_pooler(hidden_states[layer])
            else:
                pooled = torch.cat([pooled, self._mean_pooler(hidden_states[layer])], dim=1)
        
        return pooled



def get_params_to_learn(model):
    return [name for name, param in model.named_parameters() if param.requires_grad == True]

