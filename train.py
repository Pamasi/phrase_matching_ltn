import argparse, os
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CosineEmbeddingLoss
from transformers import DistilBertTokenizerFast
from tqdm import trange
from typing import Tuple
import wandb

from torchmetrics.classification import MulticlassAveragePrecision, MulticlassRecall,  MulticlassAccuracy

from util.dataset import PatentDataset, PatentCollator
from util.common import get_args_parser
from model.phrase_distil_bert import PhraseDistilBERT



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False


#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:4096"
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
    
    model = PhraseDistilBERT(args.score_level, use_qlora=args.qlora)
    model.to(args.device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.c_lr_min, max_lr=args.c_lr_max, cycle_momentum=False)

    criterion_ce = BCEWithLogitsLoss()
    
    criterion_sim = CosineEmbeddingLoss()

    # reproducibility
    torch.manual_seed(args.seed)
    
    # configure metrics
    metric_ap = MulticlassAveragePrecision(num_classes=args.score_level, average="macro", thresholds=None)
    metric_acc = MulticlassAccuracy(num_classes=args.score_level, average="macro")
    metric_ar = MulticlassRecall(num_classes=args.score_level, average='macro')



    # configure wandb 
    if args.no_track==False:
        wandb.login()
        run = wandb.init(project='phrase_matching', config=args)  



    model.train()

    for epoch in trange(args.n_epoch):
        # step
        train_acc, train_ap, train_ar, train_loss, train_loss_score, train_loss_emb = step(train_loader, model, optimizer, 
                                                                                           lr_scheduler, criterion_sim, 
                                                                                           criterion_ce, metric_ap,  metric_ar, metric_acc)
        val_acc, val_ap, val_ar, val_loss, val_loss_score, val_loss_emb = step(val_loader, model, optimizer, 
                                                                              lr_scheduler, criterion_sim, 
                                                                              criterion_ce, metric_ap,  metric_ar, metric_acc)
        dict_log =  {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_ap": train_ap,
                    "train_ar": train_ar,
                    "train_loss": train_loss,
                    "train_loss_score": train_loss_score,
                    "train_loss_emb": train_loss_emb,
                    "val_acc": val_acc,
                    "val_ap": val_ap,
                    "val_ar": val_ar,
                    "val_loss": val_loss,
                    "val_loss_score": val_loss_score,
                    "val_loss_emb":   val_loss_emb,
                }
        
        if args.no_track==False:
            wandb.log(dict_log )

            run.log_code()
        
        else:
            print(dict_log)

def step(data_loader, model, optimizer, lr_scheduler, criterion_sim, criterion_ce, metric_ap, metric_ar, metric_acc) -> Tuple[torch.Tensor]:

    tgt_list = []
    pred_list = []
    for anchors, targets, target_scores in data_loader:
        (latent1, latent2, out_scores) = model(anchors, targets)

        optimizer.zero_grad()
        
        loss_score = criterion_ce(target_scores, out_scores)
        

        label_type = torch.tensor([ 1 if torch.argmax(scores) <2 else  -1 for scores in target_scores ], dtype=torch.float32, device=target_scores.device)

        loss_emb = criterion_sim(latent1, latent2, label_type)


        loss = loss_score + loss_emb
        loss.backward()
        optimizer.step()
            
        lr_scheduler.step()

        # one-hot encoding to spare gpu memory
        tgt_list.append(torch.tensor([torch.argmax(scores) for scores in target_scores ], dtype=torch.float32, device=target_scores.device))
        pred_list.append(out_scores)

    # calculate metrics
    preds = torch.vstack(pred_list)
    tgts= torch.vstack(tgt_list)

    ap = metric_ap(preds, tgts)
    ar= metric_ar(preds, tgts)
    acc= metric_acc(preds, tgts)

    return acc, ap, ar, loss, loss_score, loss_emb

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DistilledBERT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)