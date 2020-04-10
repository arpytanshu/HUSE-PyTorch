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



pretrained_imagenet_models = {
    'resnet18'      : models.resnet18,
    'resnet101'     : models.resnet101,
    'resnet152'     : models.resnet152,
    'resnext101'    : models.resnext101_32x8d,
    'resnext50'     : models.resnext50_32x4d,
    'default'       : models.resnet18
    }



pretrained_bert_models = [  'bert-base-cased',
                            'bert-large-cased',
                            'bert-base-uncased',
                            'bert-large-uncased',
                            ]



'''
Methods for getting model for Image Embeddings
We use pre-trained models trained on ImageNet from torchvision
to get 1000-Dimentional vector representation for images.
'''
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

NOTE:
The HUSE paper described the text description for each sample (text modal)
to be much greater than 512 tokens for the dataset it used { UPMC Food-101 } 
and thus required to extract 512 most important tokens from the text description.

However the textual description of the samples in the dataset shared with us is 
much smaller than 512, thus eliminating the need for such token clipping.
'''
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



def _mean_pooler(encoding):
    return encoding.mean(dim=1)
    

    
def bert_embeddings_pooler(bert_out, num_layers_to_use):
    '''
    Concatenates embeddings from last N ( = num_layers_to_usehidden ) hidden layer states.
    The embedding for each layer is mean pooled along the time / sequence dimension.
    Example
    -------
    This concatenate the embeddings from the last num_layers_to_use layers for each token 
    and then average all token embeddings for a sequence.
        out = BertModel(input_ids)
        embeddings = bert_embeddings_pooler(out, 4)
    '''
    hidden_states = bert_out[2]
    layers_to_use = range(-num_layers_to_use, 0)    
    pooled = None
    for layer in layers_to_use:
        if pooled is None:
            pooled = _mean_pooler(hidden_states[layer])
        else:
            pooled = torch.cat([pooled, _mean_pooler(hidden_states[layer])], dim=1)
    return pooled



class HUSE_config():
    
    def __init__(self):
        self.tfidf_dim = None
        self.num_classes = None 
        self.image_embed_dim = None
        self.bert_hidden_dim = None
        self.num_bert_layers = 4 # as in HUSE paper
        self.image_tower_hidden_dim = 512 # as in HUSE paper
        self.text_tower_hidden_dim = 512 # as in HUSE paper        

        # parameters for Losses
        # ---------- --- ------
        self.alpha = 0.33 # weight for classificaion loss
        self.beta = 0.33 # weight for semantic similarity loss
        self.gamma = 0.33 # loss for cross modal loss
        self.margin = 0.8 # relaxation parameter for semantic similarity loss

    def __repr__(self):
        string = []
        for attr in dir(self):
            if attr[0] != '_':
                string.append(attr + ' : ' + str(eval('self.'+attr)))
        return '\n'.join(string)
    

    

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
    
    def __init__(self, config):
        super(HUSE, self).__init__()
        self.config = config
        
        self.ImageTower = ImageTower(config)
        self.TextTower = TextTower(config)
        self.shared_fc_layer = nn.Linear(
                in_features = config.image_tower_hidden_dim + config.text_tower_hidden_dim,
                out_features = config.num_classes)
        

    def forward(self, image_embedding, text_embedding):        
        out1 = self.ImageTower(image_embedding)
        out2 = self.TextTower(text_embedding)
        out = self.shared_fc_layer(torch.cat([out1, out2], dim=1))
        return out
    


def get_params_to_learn(model):
    params_to_learn = []
    for param in model.named_parameters():
        if param.requires_grad:
            params_to_learn.append(param)
    return params_to_learn



class ClassificationLoss(nn.Module):
    
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        
    def forward(self, UE, labels):
        '''
        UE : embeddings created from model
        target: index corresponding to class names
        '''
        loss = F.cross_entropy(UE, labels)
        return loss



class CrossModalLoss(nn.Module):
    
    def __init__(self):
        super(CrossModalLoss, self).__init__()
        
    def forward(self, image_UE, text_UE):
        '''
        image_UE: universal embeddings created using only images
        text_UE : universal embeddings created using only text
        '''
        loss = F.cosine_similarity(image_UE, text_UE).mean()
        return loss

        

class SemanticSimilarityLoss(nn.Module):
    
    def __init__(self, margin, A, device):
        '''
        A : Semantic Graph created from the class name embeddings
        margin : relaxation margin        
        '''
        super(SemanticSimilarityLoss, self).__init__()
        self.margin = margin
        self.A = A.to(device)
        
    def forward(self, UE, labels):
        '''
        Parameters
        ----------
        UE : universal embeddings for samples in batch.
        A  : tensor of size size [ num_classes x num_classes ]
            Semantic Graph of distance between embedded class names
        labels : index of sample's class name in Semantic Graph
                 must have values between 0 & num_classes
        
        Example
        -------
        batch_size = 2
        
        We have 2 universal embedding and 2 labels in this batch.
        UE = [UE_1 UE_2] # universal embeddings for batch samples
        L =  [C_1 C_2]   # index corresponding to the class names of samples
        
        To calculate pairwise_embedding_distance, we create pairs of 
        samples in batch and calc their cosine distances.
        UE_m = UE.repeat_interleave(2, -1)  # [ UE_1  UE_1  UE_2  UE_2 ]
        UE_n = UE.repeat(2,1)               # [ UE_1  UE_2  UE_1  UE_2 ]
        pairwise_embedding_distance = distance( UE_m, UE_n )
        
        Similarly pairwise label index are used to index into the Semantic Graph A
        L_m = labels.repeat_interleave(2, -1)   # [ C_1  C_1  C_2  C_2 ]
        L_n = labels.repeat(2,1)                # [ C_1  C_2  C_1  C_2 ]
        
        pairwise_class_distance = A[[L_m, L_n]]
        '''
        N = UE.shape[0] # batch size
        
        UE_m = UE.repeat_interleave(N, -2)
        UE_n = UE.repeat(N,1)
        pairwise_embedding_distance = F.cosine_similarity(UE_m, UE_n) # d (U_m , U_n )
        
        L_m = labels.repeat_interleave(N)
        L_n = labels.repeat(N)
        pairwise_class_distance = self.A[[L_m, L_n]]  # Aij

        sigma = self._calc_sigma(pairwise_class_distance, pairwise_embedding_distance)
        # loss = Σ ( σ * (d(U_m,U_n) - Aij)² ) / N²
        loss = (sigma * (pairwise_embedding_distance - pairwise_class_distance).pow(2)).mean() 
        return loss
    
    def _calc_sigma(self, pairwise_class_distance, pairwise_embedding_distance):
        # calculate sigma as described in eq(10) in the HUSE paper.
        sigma = ( pairwise_class_distance < self.margin ) & ( pairwise_embedding_distance < self.margin )
        return sigma.to(torch.float32)



class EmbeddingSpaceLoss(nn.Module):
    
    def __init__(self, A, device, config : HUSE_config):
        '''
        config : A HUSE_config object containing model configuration parameters
        Parameters
        ----------
        A     : Semantic Graph of distance between embedded class names
        alpha : weight to control influence of Classification Loss
        beta  : weight to control influence of Semantic Similarity Loss
        gamma : weight to control influence of Cross Modal Loss
        margin : relaxation margin used in Semantic Similarity Loss
        '''
        super(EmbeddingSpaceLoss, self).__init__()
        self.config = config
        self.Loss1 = ClassificationLoss()
        self.Loss2 = SemanticSimilarityLoss(self.config.margin, A, device)
        self.Loss3 = CrossModalLoss()

        
    def forward(self, UE, UE_image, UE_text, labels):
        '''
        Parameters
        ----------
        UE : UE : universal embeddings for samples in batch.
        UE_image : universal embeddings created using only image
        UE_text : universal embeddings created using only text
        labels : index corresponding to class names
        '''
        loss1 = self.config.alpha * self.Loss1(UE, labels)
        loss2 = self.config.beta  * self.Loss2(UE, labels)
        loss3 = self.config.gamma * self.Loss3(UE_image, UE_text)
        loss = loss1 + loss2 + loss3
        return loss        
