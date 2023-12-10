import torch
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast


class PatentDataset(Dataset):
    """_summary_

    Args:
        Dataset (torch.Dataset): encapsulate the Patent file into a torch Dataset 
    """

    def __init__(self, path:str, tokenizer:DistilBertTokenizerFast, max_len:int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
     
        raw_data = pd.read_csv(path, usecols=['anchor', 'target', 'context', 'score'])
        self._process(raw_data)
        
    

    def _process(self, raw_data:pd.DataFrame):
        """ fuse the anchor and context into a continous token representation
        """
        
        # save only relevant data
        raw_data['anchor_context'] = raw_data.apply(lambda row: row['anchor']+ ';'+ row['context'], axis=1)
        self.data = raw_data[['anchor_context', 'target', 'score']]
        
        
    def __len__(self):
        return len(self.text)
    

    def __getitem__(self, index):
        
        anchor_text = self.data['anchor_context'][index]
        anchor_text = " ".join(anchor_text.split())
        
        target_text = self.data['target'][index]
        target_text = " ".join(target_text.split())       


        anchor = self.tokenizer.encode_plus(
            anchor_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )        
        
        target = self.tokenizer.encode_plus(
            target_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        

        anchor_data= {
            'ids': torch.tensor(anchor['input_ids'], dtype=torch.long),
            'mask': torch.tensor(anchor['attention_mask'], dtype=torch.long),
            'token_type_id': torch.tensor(anchor['token_type_ids'], dtype=torch.long)
        }
        target_data= {
            'ids': torch.tensor(anchor['input_ids'], dtype=torch.long),
            'mask': torch.tensor(anchor['attention_mask'], dtype=torch.long),
            'token_type_id': torch.tensor(anchor['token_type_ids'], dtype=torch.long)
        }

        return {
            'anchor': anchor_data,
            'target': target_data,
            'score': torch.tensor(self.data.iloc[index, 'score'].astype(float), dtype=torch.float)
        }