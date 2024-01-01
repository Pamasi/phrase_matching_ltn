import argparse, os
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CosineEmbeddingLoss
from transformers import DistilBertTokenizerFast
from tqdm import trange
from typing import Dict, Optional, Tuple, Callable
import wandb
import random
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassRecall,  MulticlassAccuracy
from torchmetrics import Metric
import optuna
from util.dataset import PatentDataset, PatentCollator
from util.common import get_args_parser, save_ckpt
from model.phrase_distil_bert import PhraseDistilBERT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False

os.environ["WANDB_START_METHOD"] = "thread"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:4096"




class Objective:
    """object function for hyperparameters search
    """
    def __init__(self, lora_rank_range:Tuple[int],
                    lora_alpha_range:Tuple[int],
                    lr_range:Tuple[float],
                    score_weight_range: Tuple[float],
                    emb_weight_range: Tuple[float],
                    args:argparse.ArgumentParser):

        self.rank_min = lora_rank_range[0]
        self.rank_max = lora_rank_range[1]
        self.alpha_min = lora_alpha_range[0]
        self.alpha_max = lora_alpha_range[1] 
        
        self.lr_min = lr_range[0]
        self.lr_max = lr_range[1]

        self.sw_r_min = score_weight_range[0]
        self.sw_r_max = score_weight_range[1]

        self.ew_r_min = emb_weight_range[0]
        self.ew_r_max = emb_weight_range[1]

        self.args = args

    def __call__(self, trial):

        self.args.qlora = trial.suggest_int("qlora_rank", self.rank_min, self.rank_max)
        self.args.qlora_alpha = trial.suggest_int("qlora_alpha", self.alpha_min, self.alpha_max)

        self.args.emb_weight = trial.suggest_float("score_weight_loss", self.sw_r_min, self.sw_r_max)
        self.args.score_weight = trial.suggest_float("emb_weight_loss", self.ew_r_min, self.ew_r_max)
        self.args.lr = trial.suggest_float("lr", self.lr_min, self.lr_max)


      
        return experiment(args)



def experiment(args)->torch.float:
 

    train_loader, val_loader = create_loader(args)
    
    model = PhraseDistilBERT(args.score_level, use_qlora=args.qlora, qlora_rank=args.qlora_rank, qlora_alpha=args.qlora_alpha, freeze_emb=args.freeze_emb)
    model.to(args.device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.c_lr_min, max_lr=args.c_lr_max, cycle_momentum=False)

    criterion = { 'ce': BCEWithLogitsLoss(), 
                 'sim': CosineEmbeddingLoss(), 
                 'emb_weight':args.emb_weight,
                 'score_weight':args.score_weight}


    # reproducibility
    torch.manual_seed(args.seed)    
    random.seed(23)
    
    # configure metrics
    metric_ap = MulticlassAveragePrecision(num_classes=args.score_level, average="macro", thresholds=None).to(args.device)
    metric_acc = MulticlassAccuracy(num_classes=args.score_level, average="macro").to(args.device)
    metric_ar = MulticlassRecall(num_classes=args.score_level, average='macro').to(args.device)

    metric = { 'ap': metric_ap, 'ar': metric_ar, 'acc': metric_acc}

    # configure wandb 
    if args.no_track==False:
        wandb.login()

        wandb_run_name = f'SW{args.score_weight}_EW{args.emb_weight}_B{args.batch}_LR{args.lr}'

        if args.qlora:
            wandb_run_name = f'QR{args.qlora_rank}A{args.qlora_alpha}' + '_' + wandb_run_name
        
        run = wandb.init(project='phrase_matching', config=args, name=wandb_run_name,  reinit=True,)  



    model.train()

    best_ap = 0 
    for epoch in trange(args.n_epoch):
        # step
        train_loss, _= step(train_loader, args.device, model, optimizer, lr_scheduler, criterion, move_to_gpu=PatentCollator.move_to_gpu)
                                                                                           
        val_loss, val_metric   = step(val_loader, args.device, model, optimizer, lr_scheduler, criterion, move_to_gpu=PatentCollator.move_to_gpu, metric=metric)
        dict_log =  {
                    "epoch": epoch,
                    "train/loss": train_loss['tot'],
                    "train/loss_score": train_loss['score'],
                    "train/loss_emb": train_loss['emb'],
                    "val/acc": val_metric['acc'],
                    "val/ap":  val_metric['ap'],
                    "val/ar":  val_metric['ar'],
                    "val/loss": val_loss['tot'],
                    "val/loss_score": val_loss['score'],
                    "val/loss_emb":   val_loss['emb']
                }
        
        if args.no_track==False:
            wandb.log(dict_log )

            run.log_code()

        else:
            print(dict_log)

        # save best weight
        if val_metric['ap'] > best_ap:
            best_ap = val_metric['ap']
            save_ckpt(model, epoch, train_loss['tot'],
                      optimizer, lr_scheduler,args.dir, torch.get_rng_state(), save_best=True)
            
    return val_metric['ap']

def create_loader(args):
    tokenizer =  DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    
    path_train =  os.path.join(os.getcwd(), args.path_train)
    path_val =  os.path.join(os.getcwd(), args.path_val)
    
    torch.cuda.set_device(0)
    train_data = PatentDataset(path_train, tokenizer, args.max_len, args.seed, p_syn=args.p_syn)
    val_data = PatentDataset(path_val, tokenizer, args.max_len,args.seed, is_val=True)

    collate_fn = PatentCollator()

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,persistent_workers=True, drop_last=False)
                            
    return train_loader,val_loader

def step(data_loader:DataLoader, device:torch.device, model:torch.nn.Module, 
         optimizer:torch.optim, lr_scheduler:torch.optim.lr_scheduler, 
         criterion: Dict[str, torch.nn.Module], 
         move_to_gpu: Callable, metric: Optional[Dict[str, Metric]]=None) -> Tuple[Dict[str,torch.Tensor]]:
    """ execute a step of one epoch

    Args:
        data_loader (DataLoader): data loader
        device (torch.device):device to be used
        model (torch.nn.Module): neural network to train
        optimizer (torch.optim): optimizer to be used in training
        lr_scheduler (torch.optim.lr_scheduler): learning rate
        criterion (Dict[str, torch.nn.Module]): dict of the different loss to be computed
        move_to_gpu (Callable): function used to move data from cpu to gpu
        metric (Optional[Dict[str, Metric]): dict of all metrics to be computed (only for validation set). Default None

    Returns:
        Tuple[Dict[str,torch.Tensor]]: metrics and loss results
    """

    use_metric = False if metric==None else True 

    if use_metric:
        tgt_list = []
        pred_list = []

    for anchors_cpu, targets_cpu, target_scores_cpu in data_loader:
        # TODO: find a nicer way
        anchors, targets, target_scores = move_to_gpu(device, anchors_cpu, targets_cpu, target_scores_cpu)
        (latent1, latent2, out_scores) = model(anchors, targets)

        optimizer.zero_grad()
        
        loss_score = criterion['ce'](target_scores, out_scores)
        

        label_type = torch.tensor([ -1 if torch.argmax(scores) <2 else  1 for scores in target_scores ], dtype=torch.float32, device=target_scores.device)

        loss_emb = criterion['sim'](latent1, latent2, label_type)


        loss = loss_score*criterion['score_weight'] + loss_emb*criterion['emb_weight']
        loss.backward()
        optimizer.step()
            
        lr_scheduler.step()

        if use_metric:
            # one-hot encoding to spare gpu memory
            tgt_list.append(torch.tensor([torch.argmax(scores) for scores in target_scores ], dtype=torch.long, device=target_scores.device))
            pred_list.append(out_scores)

    if use_metric:
        # calculate metrics
        preds = torch.concat(pred_list)
        tgts= torch.concat(tgt_list)

        # pack results
        res_metric = {  'ap': metric['ap'](preds, tgts),
                        'ar':metric['ar'](preds, tgts),
                        'acc': metric['acc'](preds, tgts)
        }
    else:
        res_metric = None

    res_loss = {'tot': loss,
                'score':loss_score,
                'emb': loss_emb}
        
    return  res_loss, res_metric



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DistilledBERT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.optuna:
        objective = Objective(lora_rank_range=(1,8), lora_alpha_range= (1,16), lr_range=(2e-6, 2e-4),score_weight_range= (0.1, 15),emb_weight_range= (0.1,15), args=args)
        study = optuna.create_study(
            direction="maximize",
            study_name="NAS",
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=5, timeout=600)

    else:
        experiment(args)