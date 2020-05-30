# -*- coding: utf-8 -*-
"""netaporter-HUSE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ins3TW3jw4Wv4fUwgF-2xLjzvyykqhZg

# HUSE: Hierarchical Universal Semantic Embeddings

```
This is my implementation of the HUSE paper in PyTorch, as part of an assessment assignment at GreenDeck.

https://arxiv.org/pdf/1911.05978.pdf
This code is also available at: https://github.com/arpytanshu/HUSE-PyTorch

author: arpytanshu@gmail.com
```
"""

! pip install transformers

import time
import torch
import zipfile
import pandas as pd
import torch.nn.functional as F

from torch import nn
from PIL import Image
from torchvision import models
from torch.optim import RMSprop
from torchvision import transforms
from transformers import BertModel, BertConfig
from keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig

# ======== === ====== #
# NOTEBOOK RUN CONFIG #
# ======== === ====== #

DEV_MODE = False  # for development runs only
NUM_DEV_SAMPLES = 0 # number of samples to use in DEV_MODE

# IMAGE_ARCHIVE_PATH = '/content/drive/My Drive/TEMP/HUSE/val_images.zip'
# DATAFRAME_PATH = '/content/drive/My Drive/TEMP/HUSE/validation_data.csv'
# IMAGE_PATH = '/content/dataset_images/content/sample_data/val_images/'

IMAGE_PATH = '/content/dataset_images/netaporter_gb/'
IMAGE_ARCHIVE_PATH = '/content/drive/My Drive/TEMP/HUSE/netaporter_gb.zip'
DATAFRAME_PATH = '/content/drive/My Drive/TEMP/HUSE/training_data.csv'

VAL_SPLIT = 0.1
BERT_MODEL_NAME = 'bert-base-uncased'
OPTIM_LEARNING_RATE = 1.6192e-05
OPTIM_MOMENTUM = 0.9
NUM_EPOCHS = 10
BATCH_SIZE = 64 if DEV_MODE else 512


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print('Device Type: ', device.type)
if device.type=='cuda':
  dtype = torch.cuda.FloatTensor

with zipfile.ZipFile(IMAGE_ARCHIVE_PATH, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset_images/')

"""#  
# **Data**
50k products accompanied along with its image, text and class name

To apply transforms as we feed data into the model, we make a PyTorch DataSet.  
#  
#### **The train Dataset object from the can be queried later to get the following:**
- A tfidf-vectorizer fitted on all text sample. { self.tokenizer }
- Mapping from class_names to class_index
#  

#### **The Dataset reads in the image, text and labels for each sample and applies the following transformations to them:**  
- Transformations on Image
  - Resize [ to 224x224, (as expected by pre-trained torchvision models) ]
  - Random Contrast, Brightness, Hue, Saturation
  - Random Horizontal Flip
  - Normalizes each channel (as expected by pre-trained torchvision models)
- Cleans text { **!TODO** }
- Tokenizes the text and makes input_ids for getting BERT embeddings for text
- Gets the tfidf vector for text
#  

#### **The \_\_getitem__ method for the Dataset returns the following for each sample:** 
- image : torch tensor of shape 3 x 224 x 224
- bert_input_ids : torch tensor ( input ids corresponding to text tokens, created by BertTokenizer)
- tfidf_vector : torch tensor (tfidf feature vectors of shape self.tokenizer.num_words)
- labels : torch tensor ( index corresponding to the class name )
"""

class NAPDataset(Dataset):
    
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
        return 'len:{} classes:{}'.format(len(self.df), self.num_classes)

    def __len__(self):
        return self.df.__len__()


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

"""# **Semantic Graph**

To create a Semantic Graph, we take in the class_name to class_index mapping.

This mapping should be obtained from the Dataset object.
```
tr_dataset = utils.NAPDataset(tr_df.copy(),
                          image_path = IMAGE_PATH,
                          train_mode = True,
                          bert_model_name = BERT_MODEL_NAME,
                          transforms = utils.image_transforms['train'],
                          tokenizer = None)
classes_map = tr_dataset.classes_map

classes_map
{ 'accessories<belts<wide': 0,
  'accessories<books<books': 1,
 ...
  'shoes<pumps<high heel': 41,
  'shoes<pumps<mid heel': 42
 ...
}

A = SemanticGraph(classes_map)

```

### Getting Class name Embeddings  
To get the Embeddings for each class name, we use Bert Embeddings for the 3 level of hierarchy in class names, and concatenate them together.  
If a hierarchy has multiple words, its embeddings are mean pooled across the tokens.
![semantic_graph_embedding](https://raw.githubusercontent.com/arpytanshu/HUSE-PyTorch/master/resources/semantic_graph_embedding.jpg)

Once we get the Embeddings for the class_names, we fill in the Semantic Graph at
position A[i][j] with cosine distance between embeddings of class_name[i] & class_name[j].
"""

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

"""#### The following is a class to help extract Embeddings for class names and mean pool them along the tokens.  
#### It uses Pre-Trained BertModel from HuggingFace Transformers.
"""

class BertSentenceEncoder():
    def __init__(self, model_name='bert-base-uncased'):
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
        self.device =       torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
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
            input_ids = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', pad_to_max_length=True, max_length=max_length)['input_ids']
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

"""# **Pre-trained Models**

For getting Image & Text embeddings for the data, we use pre-trained models from **torchvision** and **huggingface's transformers**.

A list of model names are added in ```pretrained_imagenet_models``` and ```pretrained_bert_models```.  


We furthur define 2 methods named ```ImageEmbeddingModel``` & ```BertEmbeddingModel```.  
These methods
- Return a pretrained image model and pertrained bert model respectively.
- All parameters for both of these models are frozen.
- They return an integer ```out_dimension```, which is the size of the embedding that the model is expected to return.

These methods can be easily modified to support different models as well, without changing the rest of the code.

# <br>

**NOTE:**  
**The HUSE paper described the text description for each sample (text modal)
to be much greater than 512 tokens (which is the limit for BERT) for the dataset it used { UPMC Food-101 } and thus required to extract 512 most important tokens from the text description.**

**However the textual description of the samples in the dataset shared with us is 
much smaller than 512, thus eliminating the need for such token clipping.**
"""

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

"""The function ```bert_embeddings_pooler``` is a helper method, which concatenates bert embeddings from the specified number ( num_layers_to_use ) of layers & mean pools them along the tokens."""

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

"""# **HUSE model**

### HUSE_config:  
A configuration class to hold configuration attributes for the HUSE model.  
By default:
- Some parameters are left to None, and are populated later.  
- Some parameters have values from the HUSE paper.  
- Some parameters have a dummy value.

# <br>

### ImageTower:
The ImageTower module is modelled exactly as described in the HUSE paper.

# <br>

### TextTower:
The TextTower module is modelled exactly as described in the HUSE paper.

# <br>

### HUSE_model:
Uses instances of TextTower, ImageTower & a SharedLayer to completely model the HUSE model as described in the HUSE paper.  
This model holds all the learnable parameters.
"""

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

"""# **Losses**

The HUSE paper describes 3 loss functions that are to be used so that the Embedding space has the required properties.

###1. **Classification Loss:**  
This is the softmax cross entropy loss. Implementing this in PyTorch is a 1-liner.
"""

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

"""###2. **Cross Modal Loss:**  
This loss tries to minimize the universal embedding gap when an embedding for the same sample is created using either it's text representation, or it's image representation.  
For CrossModalLoss, we get the universal embeddings from the HUSE_model using only either image or text at once.  
- For obtaining universal embedding for text, we usa a Zero matrix { of appropriate shape } for representing the image embedding.
- For obtaining universal embedding for image, we usa a Zero matrix { of appropriate shape } for representing the text embedding.  

To get the loss, we take the cosine distance between them. Since these embeddings are coming from the same sample ( and only different modal ), minimizing the distance between them bring the embeddings from the 2 modals closer in universal embedding space.
"""

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

"""###3. **Semantic Similarity Loss:**  
For demonstrating how I implemented this loss, here is a simple example:  

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

- Once we get ```pairwise_embedding_distance``` & ```pairwise_class_distance``` (from the semantic graph), we calculate sigma using the equation:  
![alt text](https://raw.githubusercontent.com/arpytanshu/HUSE-PyTorch/master/resources/SSLoss-sigma.png)

- For getting the final loss, we again use the equation as describer in the HUSE paper.  
 loss = Σ ( σ * ( d (U_m,U_n) - Aij )² ) / N²
![alt text](https://raw.githubusercontent.com/arpytanshu/HUSE-PyTorch/master/resources/SSLoss-equation.png)
"""

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
        '''
        N = UE.shape[0] # batch size
        
        UE_m = UE.repeat_interleave(N, -2)
        UE_n = UE.repeat(N,1)
        pairwise_embedding_distance = F.cosine_similarity(UE_m, UE_n) # d (U_m , U_n )
        
        L_m = labels.repeat_interleave(N)
        L_n = labels.repeat(N)
        pairwise_class_distance = self.A[[L_m, L_n]]  # Aij

        # print(pairwise_embedding_distance.device)
        # print(pairwise_class_distance.device)
        # print(self.margin.device)
        # print(self.A.device)

        sigma = self._calc_sigma(pairwise_class_distance, pairwise_embedding_distance)
        # loss = Σ ( σ * (d(U_m,U_n) - Aij)² ) / N²
        loss = (sigma * (pairwise_embedding_distance - pairwise_class_distance).pow(2)).mean() 
        return loss

        
    def _calc_sigma(self, pairwise_class_distance, pairwise_embedding_distance):
        # calculate sigma as described in eq(10) in the HUSE paper.
        sigma = ( pairwise_class_distance < self.margin ) & ( pairwise_embedding_distance < self.margin )
        return sigma.to(torch.float32)

"""### We wrap the 3 Loss modules into another module that also weights the 3 Losses using parameters from the HUSE_config:
- HUSE_config.alpha
- HUSE_config.beta
- HUSE_config.gamma  
![alt text](https://raw.githubusercontent.com/arpytanshu/HUSE-PyTorch/master/resources/combined_loss.png)
"""

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

"""# **Train Loop**"""

# get dataframes, dataset and dataloader
# --- ----------- ------- --- ----------
tr_df = pd.read_csv(DATAFRAME_PATH)
if DEV_MODE:
    tr_df = tr_df.sample(NUM_DEV_SAMPLES)
tr_df.reset_index(drop=True, inplace=True)
tr_dataset = NAPDataset(tr_df.copy(),
                          image_path = IMAGE_PATH,
                          train_mode = True,
                          bert_model_name = BERT_MODEL_NAME,
                          transforms = image_transforms['train'],
                          tokenizer = None) # also provide tokenizer if this is a test dataset
tr_dataloader = DataLoader(tr_dataset, batch_size = BATCH_SIZE, shuffle = True)
print('*** Dataloader object created. ***')




# create Semantic Graph
# ------ -------- -----
print('*** Semantic Graph ...', end=' ')
A = SemanticGraph(tr_dataset.classes_map)
print('created. ***')




# Create image and text feature extraction models
# ------ ----- --- ---- ------- ---------- ------
Image_Embedding_Model, image_emb_dim = ImageEmbeddingModel('resnet18')
Bert_Embedding_Model, bert_emb_dim = BertEmbeddingModel(BERT_MODEL_NAME)
print('*** Image_Embedding_Model created. ***')
print('*** Bert_Embedding_Model created. ***')





# CONFIGURE dimensions in HUSE_config object
# --------- ---------- -- ----------- ------
huse_config = HUSE_config()
huse_config.image_embed_dim = image_emb_dim
huse_config.bert_hidden_dim = bert_emb_dim
huse_config.tfidf_dim       = tr_dataset.tokenizer.num_words
huse_config.num_classes     = tr_dataset.num_classes
print('*** huse_config object created. ***')






# Create HUSE model, Loss, Optimizer & set PyTorch device
# ------ ---- ------ ---- - -------- - --- ------- ------
HUSE_model = HUSE(huse_config)
EmbeddingLoss = EmbeddingSpaceLoss(A, device, huse_config)
optimizer = RMSprop(HUSE_model.parameters(), lr=OPTIM_LEARNING_RATE, momentum=OPTIM_MOMENTUM)
print('*** HUSE_model model created. ***')
print('*** EmbeddingSpaceLoss object created. ***')
print('*** optimizer created. ***')




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print('Device Type: ', device.type)
if device.type == 'cuda':
    HUSE_model.cuda()
    Bert_Embedding_Model.cuda()
    Image_Embedding_Model.cuda()
    print('*** Moved models to cuda device. ***')

def train_model(HUSE_model, dataloader, criterion, optimizer, num_epochs = 10):    
    since = time.time()
    train_loss_hist = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
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
                image_embedding = Image_Embedding_Model(image)
                bert_embedding = bert_embeddings_pooler(Bert_Embedding_Model(bert_input_ids), huse_config.num_bert_layers)
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



HUSE_model, hist = train_model(HUSE_model, tr_dataloader, EmbeddingLoss, optimizer)
