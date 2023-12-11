import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
import math
from typing import Tuple

class PhraseDistilBERT(nn.Module):
    def __init__(self, score_level:int=5, pool_out=256):
        super(PhraseDistilBERT, self).__init__()
        self.emb1= DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.emb2= DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pool =nn.AdaptiveAvgPool2d((1, pool_out))
        

        self.out_size = score_level

        self.W = nn.Parameter(torch.rand(pool_out*3, score_level, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(pool_out,128),
        #     nn.ReLU(), 
        #     nn.Linear(128,32),      
        #     nn.ReLU(),
        #     nn.Linear(32, score_level),
        #     nn.ReLU()
        #     )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, anchor, target) ->Tuple[torch.Tensor]:
        hidden1 = self.emb1(input_ids=anchor['ids'], attention_mask=anchor['mask']).last_hidden_state
        hidden2 = self.emb2(input_ids=target['ids'], attention_mask=target['mask']).last_hidden_state
  
       
        pool1 = self.pool(hidden1).squeeze(dim=1)
        pool2 = self.pool(hidden2).squeeze(dim=1)
      
        x = torch.cat((pool1, pool2, torch.abs(pool1 - pool2)), dim=-1)
        x = self.sigmoid(torch.matmul(x,self.W))
        scores = self.softmax(x)

        return (pool1, pool2, scores)

