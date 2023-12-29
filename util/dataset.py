import torch
import torch.nn.functional as F
import pandas as pd
import torch
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
# from transformers import  GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import random

class PatentDataset(Dataset):
    """dataset of Patents

    Args:
        Dataset (torch.Dataset): encapsulate the Patent file into a torch Dataset 
    """

    def __init__(self, path:str, tokenizer:DistilBertTokenizerFast, max_len:int, seed:int,
                 p_syn:float=0.5):
        """ create a PatentDataset object

        Args:
            path (str): path of the dataset
            tokenizer (DistilBertTokenizerFast): tokenizer
            max_len (int): max lenght of a sentence
            seed (int): seed of the random generator
            p_syn (float): probability used of changing POS in a phrase
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.level_score = 5
     
        # create the decoder that acts as prior knownledge
        # self.prior_tok   =   GPT2Tokenizer.from_pretrained(prior_decoder)
        # self.prior_model =   GPT2LMHeadModel.from_pretrained(prior_decoder)
        
        raw_data = pd.read_csv(path, usecols=['anchor', 'target', 'context', 'score'])
        self._process(raw_data)
        
        random.seed(seed)

        # dowload util for POS synset
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        self.p_syn=0.5
    

    def _process(self, raw_data:pd.DataFrame):
        """fuse the anchor and context into a continous token representation

        Args:
            raw_data (pd.DataFrame): dataset
        """
        
        # save only relevant data
        raw_data['anchor_context'] = raw_data.apply(lambda row: row['anchor']+ ';'+ row['context'], axis=1)
        self.data = raw_data[['anchor_context', 'target', 'score']]
        
        
    def __len__(self):
        return self.data.shape[0]
    

    def _convert_score(self,score:float)->torch.Tensor:
        """convert score into one-hot encoding

        Args:
            score (float): score level

        Returns:
            torch.Tensor: one-hot encoding vector
        """
        
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
    def __sub_synomyn(self,target_phrase:str)->str:
        """substitutes POS of a target phrase with a similar one
        Args:
            target_phrase (str): phrase to be modified

        Returns:
            str: modified phrase
        """
   
        tok_list = str(target_phrase).split()
        text = word_tokenize(target_phrase)
        pos_info = nltk.pos_tag(text)

        idx = random.randint(0,len(pos_info)-1)

        # substitute word if is adj, adv, nounm pron, verrb
        adj = ['JJ', 'JJR', 'JJS']
        noun = ['NN', 'NNS']
        verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBZ']

        w_tag = pos_info[idx][1]
        tok = pos_info[idx][0]
        if  w_tag in adj:
            synset = wn.synsets(tok, pos=wn.ADJ)
        elif w_tag in noun:
            synset = wn.synsets(tok, pos=wn.NOUN)
        elif w_tag in verb:
            synset = wn.synsets(tok, pos=wn.VERB)


        lemma = [ l.name() for s in wn.synsets('mixing', pos=wn.VERB) for l in s.lemmas() ]

        idx_lemma = random.randint(0,len(lemma)-1)

        tok_list[idx] = lemma[idx_lemma]


        target_synomym = ' '.join(tok_list)

        return target_synomym
    
    def __getitem__(self, index):
        
        score  =  self._convert_score(self.data['score'][index].astype(float))
        anchor_text = self.data['anchor_context'][index]
        anchor_text = " ".join(anchor_text.split())
        
        target_text = self.data['target'][index]
        anchor_text = " ".join(anchor_text.split())

        if random.random() > self.p_syn:
            target_text = self.__sub_synomyn(target_text)

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
        # prompt= f'{target_text} unrelated to:'
        # unrelated_output = self.prior_model.generate(self.prior_tok.encode(prompt, return_tensors="pt"), max_length=30)[0]
        
        # unrelated_text = self.prior_tok.decode(unrelated_output, skip_special_tokens=True)
        
        # unrelated_text = " ".join(unrelated_text.split())       
        
        # neg_example = self.tokenizer(
        #     unrelated_text,
        #     max_length=self.max_len,
        #     add_special_tokens=True, 
        #     truncation=True, 
        #     padding='max_length', return_tensors="pt"
        # )       
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
    def __call__(self, batch:List[Dict[str, torch.Tensor]])->Tuple[torch.Tensor]:
        """ collate batch to effiency purposes

        Args:
            batch (List[Dict[str, torch.Tensor]]): batch of elements

        Returns:
            Tuple[torch.tensor]: optimized batch
        """
        anchor_ids = torch.vstack([ b['anchor']['ids'] for b in batch ])
        anchor_mask = torch.vstack([ b['anchor']['mask'] for b in batch ])
        
        anchors = {'ids': anchor_ids, 'mask': anchor_mask}
        
        target_ids = torch.vstack([ b['target']['ids'] for b in batch ])
        target_mask = torch.vstack([ b['target']['mask'] for b in batch ])

        
        targets = {'ids': target_ids, 'mask': target_mask}
        
        scores =  torch.vstack([ b['score'] for b in batch ])
        

        return (anchors, targets, scores)
    
    def pin_memory(self, batch:List[Dict[str, torch.Tensor]]) ->Tuple[torch.Tensor]:
        """pin tensors to pinned_memory

        Args:
            batch (List[Dict[str, torch.Tensor]]): batch of elements
        Returns:
            Tuple[torch.Tensor]: pinned tensor wrapped in dictionaries
        """
        (anchors, targets, scores) = batch
        anchors_pinned = {'ids': anchors['ids'].pin_memory(), 'mask':   anchors['mask'].pin_memory()}
        targets_pinned = {'ids': targets['ids'].pin_memory(), 'mask':   targets['mask'].pin_memory()}
        scores_pinned = scores.pin_memory()

        return (anchors_pinned, targets_pinned, scores_pinned)
    
    def move_to_gpu(device:torch.device, anchors:Dict[str, torch.Tuple],
                     targets:Dict[str, torch.Tuple], target_scores:Dict[str, torch.Tuple])-> Tuple[Dict[str,torch.Tensor]]:
        """move all tensor to device 

        Args:
            device (torch.device): device to be used
            anchors (Dict[str, torch.Tuple]): anchors from the batch
            targets (Dict[str, torch.Tuple]): targets from the batch
            target_scores (Dict[str, torch.Tuple]): target scores from the batch

        Returns:
            Tuple[Dict[str,torch.Tensor]]: anchors, target and target scores that are in the gpu
        """
        anchors_gpu = {'ids': anchors['ids'].to(device), 'mask':   anchors['mask'].to(device)}
        targets_gpu = {'ids': targets['ids'].to(device), 'mask':   targets['mask'].to(device)}
        target_scores_gpu = target_scores.to(device)
        return (anchors_gpu, targets_gpu ,target_scores_gpu)
    
    