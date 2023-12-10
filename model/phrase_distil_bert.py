import torch
import torch.nn as nn
from transformers import DistilBertModel
import math
from typing import Tuple

class PhraseDistilBERT(nn.Module):
    def __init__(self, score_level:int=5, pool_kernel:int=3, pool_stride:int=3):
        super(PhraseDistilBERT, self).__init__()
        self.emb1= DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.emb2= DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)


        self.classifier = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(), 
            nn.Linear(128,32),      
            nn.ReLU(),
            nn.Linear(32, score_level),
            nn.ReLU()
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, anchor, target) ->Tuple[torch.Tensor]:
        hidden1 = self.emb1(input_ids=anchor['ids'], attention_mask=anchor['mask']).last_hidden_state
      
        hidden2 = self.emb2(input_ids=target['ids'], attention_mask=target['mask']).last_hidden_state
    
       
        pool1 = self.pool(hidden1)
        pool2 = self.pool(hidden2)
      
        x = torch.cat((pool1, pool2,torch.abs(pool1-pool2)))
        x = self.classifier(x)
        out = self.softmax(x)
        
        return (hidden1, hidden2, out)

