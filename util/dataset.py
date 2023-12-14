import torch
import torch.nn.functional as F
import pandas as pd
import torch
from typing import Tuple
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from transformers import AutoTokenizer, pipeline

class PatentDataset(Dataset):
    """_summary_

    Args:
        Dataset (torch.Dataset): encapsulate the Patent file into a torch Dataset 
    """

    def __init__(self, path:str, tokenizer:DistilBertTokenizerFast, max_len:int, device:torch.device='cuda', prior_decoder:str='meta-llama/Llama-2-7b-chat-hf'):
        """ create a PatentDataset object

        Args:
            path (str): path of the dataset
            tokenizer (DistilBertTokenizerFast): tokenizer
            max_len (int): max lenght of a sentence
            prior_decoder (str): name of the decoder used as prior
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.level_score = 5
     
        # create the decoder that acts as prior knownledge
        self.prior_tok = AutoTokenizer.from_pretrained(prior_decoder)
        self.prior_pipeline = pipeline(
            "text-generation",
            model=prior_decoder,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        raw_data = pd.read_csv(path, usecols=['anchor', 'target', 'context', 'score'])
        self._process(raw_data)
        
    

    def _process(self, raw_data:pd.DataFrame):
        """ fuse the anchor and context into a continous token representation
        """
        
        # save only relevant data
        raw_data['anchor_context'] = raw_data.apply(lambda row: row['anchor']+ ';'+ row['context'], axis=1)
        self.data = raw_data[['anchor_context', 'target', 'score']]
        
        
    def __len__(self):
        return self.data.shape[0]
    

    def _convert_score(self,score:float)->torch.Tensor:
        
        cls = torch.zeros((self.level_score), )
        match score:
            case 0.0:
                idx = 0
                
            case 0.25:
                idx = 1    
            
            case 0.5:
                idx = 2
                 
            case 0.75:
                idx = 3
            
            case _:
                idx = 4

        
        cls[idx]=1.0
        return cls 
                
    def __getitem__(self, index):
        
        score  =  self._convert_score(self.data['score'][index].astype(float))
        anchor_text = self.data['anchor_context'][index]
        anchor_text = " ".join(anchor_text.split())
        
        target_text = self.data['target'][index]
        target_text = " ".join(target_text.split())       


        anchor = self.tokenizer.encode_plus(
            anchor_text,
            max_length=self.max_len,
            add_special_tokens=True, 
            truncation=True, 
            padding='max_length', return_tensors="pt"
        )      
        
        target = self.tokenizer(
            target_text,
            max_length=self.max_len,
            add_special_tokens=True, 
            truncation=True, 
            padding='max_length', return_tensors="pt"
        )
        
        
        # prompt 
        prompt = f'Given the phrase \"{target_text}\" given a phrase which is similar, but unrelated'
        unrelated_text =pipeline(
            'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.prior_tok.eos_token_id,
            max_length=self.max_len
        )
        
        unrelated_text = " ".join(unrelated_text.split())       
        
        neg_example = self.tokenizer(
            unrelated_text,
            max_length=self.max_len,
            add_special_tokens=True, 
            truncation=True, 
            padding='max_length', return_tensors="pt"
        )       
        anchor_data= {
            'ids': anchor['input_ids'],
            'mask':anchor['attention_mask'] 
        }
        target_data= {
            'ids':target['input_ids'],
            'mask':target['attention_mask']
        }


        return {
            'anchor': anchor_data,
            'target': target_data,
            'score': score
        }
        
class PatentCollator(object):
    def __init__(self, device):
        self.device = device
        
    def __call__(self, batch)->Tuple[torch.Tensor]:
        """ collate batch to effiency purposes

        Args:
            batch (_type_): batch of element

        Returns:
            Tuple[torch.tensor]: optimized batch
        """
        anchor_ids = torch.vstack([ b['anchor']['ids'] for b in batch ]).to(self.device)
        anchor_mask = torch.vstack([ b['anchor']['mask'] for b in batch ]).to(self.device)
        
        anchors = {'ids': anchor_ids, 'mask': anchor_mask}
        
        target_ids = torch.vstack([ b['target']['ids'] for b in batch ]).to(self.device)
        target_mask = torch.vstack([ b['target']['mask'] for b in batch ]).to(self.device)

        
        targets = {'ids': target_ids, 'mask': target_mask}
        
        scores =  torch.vstack([ b['score'] for b in batch ]).to(self.device)
        

        return (anchors, targets, scores)


    
    