import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, ElectraModel, AlbertModel
from peft import LoraConfig, get_peft_model, PeftModel


import math
from typing import Dict, Tuple


def qlora_mode(model_name:str, base_model:nn.Module, rank:int, alpha:int) -> PeftModel:
    if model_name.find('distilbert')>=0:
        target_modules=['q_lin', 'k_lin', 'v_lin']
    elif model_name.find('electra')>=0 or model_name.find('albert')>=0:
        target_modules=['query', 'key', 'value']
    else:
        raise ValueError(f'{model_name} is an invalid name')
    
    # set 4bit quantization
    lora_config = LoraConfig(r=rank, 
                             lora_alpha=alpha, 
                            target_modules=target_modules,
                            init_lora_weights="loftq")
    
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model

class PhraseEncoder(nn.Module):
    def __init__(self, model_name:str, bos_token:torch.Tensor, score_level:int=5, pool_out:int=256, use_qlora:bool=True, 
                 qlora_rank:int=1, qlora_alpha:int=1, freeze_emb:bool=False, use_mlp:bool=True, 
                 use_gru:bool=False, n_state:int=10):
        """_summary_
        Args:
            model_name (str): name of the model
            bos_token(torch.Tensor): Begin Of Sentence token to be used with the GRU Decoder.
            score_level (int, optional): number of score . Defaults to 5.
            pool_out (int, optional): pooling output dimension of the hidden layer from bert. Defaults to 256.
            use_qlora (bool, optional): use QLoRa. Defaults to True.
            qlora_rank (int, optional): rank of QLoRA. Defaults to 1.
            qlora_alpha (int, optional): gain of QLoRA. Defaults to 1.
            freeze_emb (int, optional): freeze embedding. Defaults to False
            use_mlp (bool, optional): use a MLP instead of a non-linear matrix projection. Defaults to True
            use_gru (bool, optional): use a GRU Decoder instead of a non-linear matrix projection. Defaults to False
            n_state (int, optional): number of state of the GRU Decoder. Defaults 10.
            
        """

        super(PhraseEncoder, self).__init__()

        if model_name.find('distilbert')>=0:
            model = DistilBertModel
        elif model_name.find('electra')>=0:
            model = ElectraModel
        elif model_name.find('albert')>=0:
            model = AlbertModel
        else:
            raise ValueError(f'{model_name} is unvalid model name')
        if use_qlora:
            print(f'QLORA enabled:\trank={qlora_rank}\talpha={qlora_alpha}')
            self.emb1= qlora_mode(model_name,model.from_pretrained(model_name), qlora_rank, qlora_alpha)
            self.emb2= qlora_mode(model_name, model.from_pretrained(model_name), qlora_rank, qlora_alpha)

        else:
            self.emb1= model.from_pretrained(model_name)
            self.emb2= model.from_pretrained(model_name)


        if freeze_emb:
            for param in self.emb2.embeddings.parameters():
                param.requires_grad = False

            for param in self.emb2.embeddings.parameters():
                param.requires_grad = False



        self.pool =nn.AdaptiveAvgPool2d((1, pool_out))
        

        self.n_state = None

        self.use_mlp = use_mlp
        self.use_gru = use_gru
        if self.use_mlp==True:
            self.mlp = nn.Sequential(
                    nn.Linear(pool_out*3,128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(), 
                    nn.Dropout(p=0.2),
                    nn.Linear(128,64),  
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64,32),  
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, score_level),
                    nn.ReLU()
            )
        elif self.use_gru:
            self.n_state  = n_state
            hidden_size = pool_out*3
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.proj = nn.Linear(hidden_size, score_level)

        else:
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

        if self.use_mlp:
            x = self.mlp(x)

        elif self.use_gru:
            decoder_input = x
            decoder_hidden = x
   
            for _ in range(self.n_state):
                tmp_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
                decoder_output = self.proj(tmp_output)
               
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  

            x = topi
        else:
            x = self.sigmoid(torch.matmul(x,self.W))

        scores = self.softmax(x)

        return (pool1, pool2, scores)

