import argparse, os, sys, logging
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CosineEmbeddingLoss, L1Loss
from flash.core.optimizers import LAMB
from transformers import DistilBertTokenizerFast, ElectraTokenizerFast, AlbertTokenizerFast,get_linear_schedule_with_warmup
from tqdm import trange
from typing import Dict, Optional, Tuple, Callable
import wandb
import random
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassRecall,  MulticlassAccuracy
from torchmetrics import Metric
import optuna
from util.dataset import PatentDataset, PatentCollator
from util.common import get_args_parser, save_ckpt
from model.phrase_encoder import PhraseEncoder

import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False

os.environ["WANDB_START_METHOD"] = "thread"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:4096"




class Objective:
    """object function for hyperparameters search
    """
    def __init__(self, score_weight_range: Tuple[float],
                    emb_weight_range: Tuple[float],
                    args:argparse.ArgumentParser):


        self.sw_r_min = score_weight_range[0]
        self.sw_r_max = score_weight_range[1]

        self.ew_r_min = emb_weight_range[0]
        self.ew_r_max = emb_weight_range[1]

        self.args = args

    def __call__(self, trial):


        self.args.emb_weight = trial.suggest_int("score_weight_loss", self.sw_r_min, self.sw_r_max)
        self.args.score_weight = trial.suggest_int("emb_weight_loss", self.ew_r_min, self.ew_r_max)


      
        return experiment(args, trial=trial)



def experiment(args, trial:Optional[optuna.Trial]=None)->torch.float:
 

    train_loader, val_loader = create_loader(args)
    
    model = PhraseEncoder(args.model_name, args.score_level, use_qlora=args.qlora, qlora_rank=args.qlora_rank, qlora_alpha=args.qlora_alpha, 
                             freeze_emb=args.freeze_emb, use_mlp=args.use_mlp)
    model.to(args.device)

    if args.use_sgd:
        optimizer = torch.optim.SGD(params =  model.parameters(), lr=args.lr)
    elif args.use_lamb:
        optimizer = LAMB(model.parameters(), lr=args.lr, eps=1e-12)
    else:
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr, eps=1e-12)

    if args.lr_range_test:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step*2 )
        loss_step = []
    elif args.use_linear_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=args.n_epoch*args.step_epoch)
    else:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.c_lr_max, 
                                                         epochs=args.n_epoch, steps_per_epoch=args.step_epoch)

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

        wandb_run_name = f'BCE_SW{args.score_weight}_EW{args.emb_weight}_B{args.batch}_LR{args.lr}'


        if args.qlora:
            wandb_run_name = f'QR{args.qlora_rank}A{args.qlora_alpha}' + '_' + wandb_run_name
        

                    
        if args.model_name.find('distilbert')>=0:
            wandb_run_name = wandb_run_name   + '_DISTILBERT'
        elif args.model_name.find('electra')>=0:
            wandb_run_name = wandb_run_name   + '_ELECTRA'
        elif args.model_name.find('albert')>=0:
            wandb_run_name = wandb_run_name   + '_ALBERT'
        if args.use_mlp:
            wandb_run_name = wandb_run_name   + '_MLP'
        elif args.use_gru:
            wandb_run_name = wandb_run_name   + '_GRU'
        if args.freeze_emb:
            wandb_run_name = wandb_run_name   + '_FREEZE_EMB'

        if args.use_sgd:
            wandb_run_name = wandb_run_name   + '_SGD'
        elif args.use_lamb:
            wandb_run_name = wandb_run_name   + '_LAMB'
        else:
            wandb_run_name = wandb_run_name   + '_ADAM'


        if args.lr_range_test:
            wandb_run_name = wandb_run_name   + '_LR_RANGE_TEST'
        
        run = wandb.init(project='phrase_matching', config=args, name=wandb_run_name,  reinit=True)  

        # automate the name folder
        args.dir = str(wandb_run_name).replace(".", "_").replace('-','m')



    best_ap = 0 
        
    if args.lr_range_test:
        it =0 
        for epoch in trange(args.n_epoch):
            it = lr_range_test(it, train_loader, val_loader, args.device, model, optimizer, lr_scheduler,
                                criterion, move_to_gpu=PatentCollator.move_to_gpu, run=run, metric=metric)
        # plt.plot(range(len(loss_step)), loss_step)
        # plt.title('LR RangeTest')
        # plt.xlabel('Step')
        # plt.ylabel('Learning Rate')

        # plt.savefig(f'{args.dir}/lr_range_test.jpg')
  
    else:
        for epoch in trange(args.n_epoch):
            # step
            train_loss= train_step(train_loader, args.device, model, lr_scheduler, optimizer, 
                                criterion, move_to_gpu=PatentCollator.move_to_gpu, max_norm=args.clip_norm)
                                                                                            
            val_loss, val_metric   = val_step(val_loader, args.device, model, 
                                        criterion, move_to_gpu=PatentCollator.move_to_gpu, metric=metric)
            dict_log =  {
                        "lr": lr_scheduler.get_last_lr()[0],
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
            if args.lr_range_test:
                loss_step.extend(train_loss['loss4step'])

                for val in val_loss['loss4step']:
                    wandb.log({'val/step':val}  )
            elif val_metric['ap'] > best_ap:
                    best_ap = val_metric['ap']
                    save_ckpt(model, epoch, train_loss['tot'],
                            optimizer, lr_scheduler,args.dir, torch.get_rng_state(), save_best=True)
                

            # update scheduler
            lr_scheduler.step()

            if args.use_optuna:
                trial.report(val_metric['acc'], epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return val_metric['acc']

def create_loader(args):
    print(args.model_name)
    if args.model_name.find('distilbert')>=0:
        tokenizer =  DistilBertTokenizerFast.from_pretrained(args.model_name, truncation=True, do_lower_case=True)
    elif args.model_name.find('electra')>0:
        tokenizer =  ElectraTokenizerFast.from_pretrained(args.model_name, truncation=True, do_lower_case=True)
    elif args.model_name.find('albert')>=0:
        tokenizer =  AlbertTokenizerFast.from_pretrained(args.model_name, truncation=True, do_lower_case=True)

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

def lr_range_test(it:int, train_loader:DataLoader,  val_loader:DataLoader, device:torch.device, model:torch.nn.Module, 
         optimizer:torch.optim, lr_scheduler:torch.optim.lr_scheduler, 
         criterion: Dict[str, torch.nn.Module], 
         move_to_gpu: Callable, metric: Optional[Dict[str, Metric]]=None,
         run:Optional[wandb.run]=None) -> int:
    """ execute a step of one epoch

    Args:
        it (int): iteration
        train_loader (DataLoader): train data loader
        val_loader (DataLoader): val data loader
        device (torch.device):device to be used
        model (torch.nn.Module): neural network to train
        optimizer (torch.optim): optimizer to be used in training
        lr_scheduler (torch.optim.lr_scheduler): learning rate used only for LR Range Test
        criterion (Dict[str, torch.nn.Module]): dict of the different loss to be computed
        move_to_gpu (Callable): function used to move data from cpu to gpu
        metric (Optional[Dict[str, Metric]): dict of all metrics to be computed (only for validation set). Default None
        wandb_run(Optional[wandb.run]): wandb_run. Default None

    Returns:
        int: last iteration
    """
    


    for anchors_cpu, targets_cpu, target_scores_cpu in train_loader:
        anchors, targets, target_scores = move_to_gpu(device, anchors_cpu, targets_cpu, target_scores_cpu)
        (latent1, latent2, out_scores) = model(anchors, targets)

        optimizer.zero_grad()
        
        train_loss_score = criterion['ce'](out_scores, target_scores)
        

        label_type = torch.tensor([ -1 if torch.argmax(scores) <2 else  1 for scores in target_scores ], dtype=torch.float32, device=target_scores.device)

        train_loss_emb = criterion['sim'](latent1, latent2, label_type)


        train_loss_tot = train_loss_score*criterion['score_weight'] + train_loss_emb*criterion['emb_weight']

        batch_loss += train_loss_tot
        train_loss_tot.backward()
        optimizer.step()
            
       
        
        val_loss, val_metric   = val_step(val_loader, args.device, model,  criterion, move_to_gpu=PatentCollator.move_to_gpu, metric=metric)
        
        dict_log =  {
                    "lr": lr_scheduler.get_last_lr()[0],
                    "train/loss": train_loss_tot,
                    "train/loss_score": train_loss_score,
                    "train/loss_emb": train_loss_emb,
                    "val/acc": val_metric['acc'],
                    "val/ap":  val_metric['ap'],
                    "val/ar":  val_metric['ar'],
                    "val/loss": val_loss['tot'],
                    "val/loss_score": val_loss['score'],
                    "val/loss_emb":   val_loss['emb']
                }
        
        it+=1
       

        wandb.log(dict_log )

        run.log_code()



        
    return   it


def val_step(data_loader:DataLoader, device:torch.device, model:torch.nn.Module, 
         criterion: Dict[str, torch.nn.Module], 
         move_to_gpu: Callable, metric: Dict[str, Metric]=None) -> Tuple[Dict[str,torch.Tensor]]:
    """ execute a step of one epoch

    Args:
        data_loader (DataLoader): data loader
        device (torch.device):device to be used
        model (torch.nn.Module): neural network to train
        optimizer (torch.optim): optimizer to be used in training
        criterion (Dict[str, torch.nn.Module]): dict of the different loss to be computed
        move_to_gpu (Callable): function used to move data from cpu to gpu
        metric ([Dict[str, Metric]): dict of all metrics to be computed.

    Returns:
        Tuple[Dict[str,torch.Tensor]]: metrics and loss results
    """
    tgt_list = []
    pred_list = []
    batch_loss_tot = 0.0
    batch_loss_score = 0.0
    batch_loss_emb = 0.0
    n_batch = len(data_loader)

    model.eval() 
    # to reduce memory footprint
    with torch.no_grad():
        for anchors_cpu, targets_cpu, target_scores_cpu in data_loader:
            anchors, targets, target_scores = move_to_gpu(device, anchors_cpu, targets_cpu, target_scores_cpu)
           
            (latent1, latent2, out_scores) = model(anchors, targets)

            loss_score = criterion['ce'](out_scores, target_scores)
            batch_loss_score += loss_score

            label_type = torch.tensor([ -1 if torch.argmax(scores) <2 else  1 for scores in target_scores ], dtype=torch.float32, device=target_scores.device)

            loss_emb = criterion['sim'](latent1, latent2, label_type)
            batch_loss_emb += loss_emb


            loss = loss_score*criterion['score_weight'] + loss_emb*criterion['emb_weight']
            batch_loss_tot += loss
                

            
            # one-hot encoding to spare gpu memory
            tgt_list.append(torch.tensor([torch.argmax(scores) for scores in target_scores ], dtype=torch.long, device=target_scores.device))
            pred_list.append(out_scores)
        
    # calculate metrics
    preds = torch.concat(pred_list)
    tgts= torch.concat(tgt_list)

    # pack results
    res_metric = {  'ap': metric['ap'](preds, tgts),
                    'ar':metric['ar'](preds, tgts),
                    'acc': metric['acc'](preds, tgts)
    }

    res_loss = {'tot': batch_loss_tot/n_batch,
                'score':batch_loss_score/n_batch,
                'emb': batch_loss_emb/n_batch
            }
        
    return  res_loss, res_metric


def train_step(data_loader:DataLoader, device:torch.device, model:torch.nn.Module, 
               lr_scheduler:torch.optim.lr_scheduler,
         optimizer:torch.optim, criterion: Dict[str, torch.nn.Module], 
         move_to_gpu: Callable, max_norm:float=0.1) -> Dict[str,torch.Tensor]:
    """ execute a step of one epoch

    Args:
        data_loader (DataLoader): data loader
        device (torch.device):device to be used
        model (torch.nn.Module): neural network to train
        optimizer (torch.optim): optimizer to be used in training
        criterion (Dict[str, torch.nn.Module]): dict of the different loss to be computed
        move_to_gpu (Callable): function used to move data from cpu to gpu
        metric (Optional[Dict[str, Metric]): dict of all metrics to be computed (only for validation set). Default None

    Returns:
        Dict[str,torch.Tensor]:  loss results
    """
    
    batch_loss_tot = 0.0
    batch_loss_score = 0.0
    batch_loss_emb = 0.0
    n_batch = len(data_loader)

    model.train()


    for anchors_cpu, targets_cpu, target_scores_cpu in data_loader:
        anchors, targets, target_scores = move_to_gpu(device, anchors_cpu, targets_cpu, target_scores_cpu)
        (latent1, latent2, out_scores) = model(anchors, targets)

        optimizer.zero_grad()
        
        loss_score = criterion['ce'](out_scores, target_scores )
        batch_loss_score += loss_score


        

        label_type = torch.tensor([ -1 if torch.argmax(scores) <2 else  1 for scores in target_scores ], dtype=torch.float32, device=target_scores.device)

        loss_emb = criterion['sim'](latent1, latent2, label_type)
        batch_loss_emb += loss_emb

        loss = loss_score*criterion['score_weight'] + loss_emb*criterion['emb_weight']
        batch_loss_tot += loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()


    res_loss = {'tot': batch_loss_tot/n_batch,
                'score':batch_loss_score/n_batch,
                'emb': batch_loss_emb/n_batch
            }
        
    return  res_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DistilledBERT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.use_optuna:

        objective = Objective(score_weight_range= (args.sw_low_bound, args.sw_high_bound),emb_weight_range= (args.ew_low_bound, args.ew_high_bound), args=args)
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        
        study = optuna.create_study(
            direction="maximize",
            study_name="Weights' losses",
            sampler=optuna.samplers.RandomSampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        
        study.optimize(objective, n_trials=args.optuna_trial)

    else:
        experiment(args)