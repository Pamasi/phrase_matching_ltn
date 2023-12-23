import argparse, os, tqdm
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from util.dataset import PatentDataset, PatentCollator
from util.common import get_args_parser
from model.phrase_distil_bert import PhraseDistilBERT
from tqdm import trange
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:4096"
def main(args):
    tokenizer =  DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    
    path_train =  os.path.join(os.getcwd(), args.path_train)
    path_val =  os.path.join(os.getcwd(), args.path_val)
    
    torch.cuda.set_device(0)
    train_data = PatentDataset(path_train, tokenizer, args.max_len)
    val_data = PatentDataset(path_val, tokenizer, args.max_len)

    collate_fn = PatentCollator(args.device)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    
    model = PhraseDistilBERT(args.score_level)
    model.to(args.device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.c_lr_min, max_lr=args.c_lr_max, cycle_momentum=False)

    criterion_cl = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    criterion_ce = torch.nn.BCEWithLogitsLoss()
    
    # reproducibility
    torch.manual_seed(args.seed)
    
    model.train()

    for epoch in trange(args.n_epoch):
        # step
        step(train_loader, model, optimizer, lr_scheduler, criterion_cl, criterion_ce)
        step(val_loader, model, optimizer, lr_scheduler, criterion_cl, criterion_ce)


def step(data_loader, model, optimizer, lr_scheduler, criterion_cl, criterion_ce):
   
    for anchors, targets, target_scores in data_loader:
        (latent1, latent2, out_scores) = model(anchors, targets)

        optimizer.zero_grad()
        
        loss_score = criterion_ce(target_scores, out_scores)
        
        # contrastive learning
        # for all equal anchors id
        # set as positive the target which have score>0
        # set as negative the targer which have score==0
        
        anchor_idx = torch.tensor([ [  (a_j==a_i).all()  for a_i in anchors['ids'] ] for a_j in anchors['ids']  ], dtype=torch.bool, device=target_scores.device)
        label_score = torch.argmax(target_scores, dim=-1)

        # set as positive the target which have score>0
        pos_ex_idx =  torch.vstack([   torch.logical_and(label_score>0, i)  for i in anchor_idx ])
        # set as negative the targer which have score==0
        # TODO: create a smart way for negative example
        neg_ex_idx =  ~pos_ex_idx 
        
        # sanity check to control whether vector is correctly constructed
        assert torch.sum(pos_ex_idx[label_score==0,label_score==0])==0.0, 'check the implementation of target vectors'
        
        
        has_pos = False
        loss_emb = 0.0
        for i in range(anchor_idx.size(0)):
            # check that there is a least a negative and positive example
            if torch.sum(pos_ex_idx[i])>0:
                loss_emb +=  criterion_cl(latent1[anchor_idx[i]], latent2[pos_ex_idx[i]], latent2[neg_ex_idx[i]])  
                
                has_pos= True

        # TODO:investigate a smarter way to account for the lack of pos example
        loss = loss_emb + loss_score if has_pos else loss_emb
        loss.backward()
        optimizer.step()
            
        lr_scheduler.step()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DistilledBERT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)