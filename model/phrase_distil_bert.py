import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from peft import LoraConfig, get_peft_model, PeftModel


import math
from typing import Dict, Tuple


def qlora_mode(base_model:nn.Module, rank:int, alpha:int) -> PeftModel:
           # set 4bit quantization
    lora_config = LoraConfig(r=rank, 
                             lora_alpha=alpha, 
                            target_modules=['q_lin', 'k_lin', 'v_lin'],
                            init_lora_weights="loftq")
    
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model

class PhraseDistilBERT(nn.Module):
    def __init__(self, score_level:int=5, pool_out:int=256, use_qlora:bool=True, 
                 qlora_rank:int=1, qlora_alpha:int=1, freeze_emb:bool=False):
        """_summary_
        Args:
            score_level (int, optional): number of score . Defaults to 5.
            pool_out (int, optional): pooling output dimension of the hidden layer from bert. Defaults to 256.
            use_qlora (bool, optional): use QLoRa. Defaults to True.
            qlora_rank (int, optional): rank of QLoRA. Defaults to 1.
            qlora_alpha (int, optional): gain of QLoRA. Defaults to 1.
            freeze_emb (int, optional): freeze embedding. Defaults to False
        """

        super(PhraseDistilBERT, self).__init__()

        if use_qlora:
            print(f'QLORA enabled:\trank={qlora_rank}\talpha={qlora_alpha}')
            self.emb1= qlora_mode(DistilBertModel.from_pretrained("distilbert-base-uncased"), qlora_rank, qlora_alpha)
            self.emb2= qlora_mode(DistilBertModel.from_pretrained("distilbert-base-uncased"), qlora_rank, qlora_alpha)

        else:
            self.emb1= DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.emb2= DistilBertModel.from_pretrained("distilbert-base-uncased")


        if freeze_emb:
            for param in self.emb2.embeddings.parameters():
                param.requires_grad = False

            for param in self.emb2.embeddings.parameters():
                param.requires_grad = False



        self.pool =nn.AdaptiveAvgPool2d((1, pool_out))
        

        self.out_size = score_level

        self.W = nn.Parameter(torch.rand(pool_out*3, score_level, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, anchor:Dict[str, torch.Tensor], target:Dict[str, torch.Tensor]) ->Tuple[torch.Tensor]:
        """_summary_

        Args:
            anchor (torch.Tensor): anchor dictionary containts the ids and mask of the input tokens
            target (torch.Tensor): target dictionary containts the ids and mask of the input tokens

        Returns:
            Tuple[torch.Tensor]: (latent embedding of the anchor,
                                  latent embedding of the target,
                                  score logits)
        """
        hidden1 = self.emb1(input_ids=anchor['ids'], attention_mask=anchor['mask']).last_hidden_state
        hidden2 = self.emb2(input_ids=target['ids'], attention_mask=target['mask']).last_hidden_state
  
       
        pool1 = self.pool(hidden1).squeeze(dim=1)
        pool2 = self.pool(hidden2).squeeze(dim=1)
      
        x = torch.cat((pool1, pool2, torch.abs(pool1 - pool2)), dim=-1)
        x = self.sigmoid(torch.matmul(x,self.W))
        scores = self.softmax(x)

        return (pool1, pool2, scores)

