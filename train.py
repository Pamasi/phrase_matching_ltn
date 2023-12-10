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
    criterion_ce = torch.nn.CrossEntropyLoss()
    
    # reproducibility
    torch.manual_seed(args.seed)
    
    model.train()
    
    for epoch in trange(args.n_epoch):
        # step
        step(train_loader, model, optimizer, lr_scheduler, criterion_cl, criterion_ce)
        step(val_loader, model, optimizer, lr_scheduler, criterion_cl, criterion_ce)


def step(data_loader, model, optimizer, lr_scheduler, criterion_cl, criterion_ce):
    for anchors, targets, scores in data_loader:
        (latent1, latent2, out_scores) = model(anchors, targets)

        optimizer.zero_grad()
        
        loss_score = criterion_ce(scores, out_scores)
        
        # contrastive learning
        # for all equal anchors id
        # set as positive the target which have score>0
        # set as negative the targer which have score==0
        
        idx_anchor = torch.tensor([ [ 1 if a_i==a_j else 0 for a_j in  anchors['ids'] ] for a_i in anchors['ids']  ], dtype=torch.bool)
        
        pos_ex_idx =  torch.tensor([ 1 if scores[a_i]>0 else 0 for a_i in idx_anchor ], dtype=torch.bool)
        neg_ex_idx =  torch.tensor([ 1 if scores[a_i]==0 else 0 for a_i in idx_anchor ], dtype=torch.bool)
        
    
        loss_emb = torch.sum( [ criterion_cl(latent1[a], latent2[p], latent2[n])   for a,p,n in zip(idx_anchor, pos_ex_idx, neg_ex_idx) ])

        loss = loss_emb + loss_score
        loss.backward()
        optimizer.step()
            
        lr_scheduler.step()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DistilledBERT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)