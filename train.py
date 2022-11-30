import torch
import argparse, yaml
import source.utils.utils_sr as utils_sr
import os
from tqdm import tqdm
import sys
import time
import numpy as np
from torch.cuda import amp
import gc

from source.dataset.dataset import get_dataset
from source.model.model import get_model
from source.optimizer import get_optimizer, get_scheduler
from source.utils.config import set_seed
from source.losses import get_criterion
from source.utils.utils import AverageMeter
from source.utils.image_utils import save_img
from source.utils.metrics import runningScore
import source.utils.utils_sr
from statistics import mean, median

import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

#Wandb and mlflow
import wandb

try:
    wandb.login(key='0b0a03cb580e75ef44b4dff7f6f16ce9cfa8a290')
    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')

def train_one_epoch(cfg, model, optimizer, scheduler, criterion, dataloader, device, epoch, stat_dict, run_log_wandb):
    model.train()
    
    scaler = amp.GradScaler()
    max_norm = 5.0
    
    
    stat_dict['epochs'] = epoch
    
    for _dl in dataloader:
        name = _dl['name']
        dataloader = _dl['dataloader']
        
        epoch_loss = AverageMeter()
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train {}:'.format(name))
        for step, (data) in pbar:

            _image_patch, _gt_patch, _idx = data
            _image_patch, _gt_patch = _image_patch.to(device), _gt_patch.to(device)
                
            batch_size = _image_patch.size(0)
            
            if cfg.train_config.mixed_pred:
                ###Use Unscaled Gradiendts instead of 
                ### https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                with amp.autocast(enabled=True):
                    _pred = model(_image_patch)
                    _pred = torch.clamp(_pred, 0, 1)
                    loss   = criterion(_pred, _gt_patch)    
                scaler.scale(loss).backward()
                
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
            else: 
                _pred = model(_image_patch)
                loss = criterion(_pred, _gt_patch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            if scheduler is not None:
                scheduler.step()
                
            epoch_loss.update(loss.item(), batch_size)
            
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix(epoch=f'{epoch}',
                            train_loss=f'{epoch_loss.avg:0.4f}',
                            lr=f'{current_lr:0.5f}', 
                            gpu_mem=f'{mem:0.2f} GB')
            
            if cfg.train_config.img_save_val and step < 10: 
                pred = _pred.data.max(1)[1].cpu().numpy()
                gt = _gt_patch.data.cpu().numpy() 
                img = _image_patch.data.cpu().numpy() 
                
                dirPath = "./result_image/train/{}/".format(name)
                fname = str(step).zfill(4)
                
                if cfg.train_config.colors == 1:
                    _img = img[0]
                    _img = np.concatenate((_img,_img,_img), axis=0).transpose(1,2,0) * 255
                else: 
                    _img = img[0].transpose(1,2,0)*255
                
                
                _pred = pred[0]
                _pred = np.expand_dims(_pred,axis=0)
                _pred = np.concatenate((_pred,_pred,_pred), axis=0).transpose(1,2,0)
                
                _gt = gt[0]
                _gt = np.expand_dims(_gt,axis=0)
                _gt = np.concatenate((_gt,_gt,_gt), axis=0).transpose(1,2,0)
                
                _pred[_pred == 0] = 0
                _pred[_pred == 1] = 255
                
                _gt[_gt == 0] = 0
                _gt[_gt == 1] = 128
                
                temp0 = np.concatenate((_gt, _img, _pred),axis=1)
                save_img(os.path.join(dirPath, str(epoch), fname + '.jpg'),temp0.astype(np.uint8), color_domain='rgb')
                
            #if cfg.train_config.debug and step < 10:
    
    if cfg.train_config.wandb:
        # Log the metrics
        wandb.log({"train/Loss": epoch_loss.avg,  
               "train/LR":scheduler.get_last_lr()[0]})
        
    stat_dict['losses'].append(epoch_loss.avg)
    return epoch_loss.avg, stat_dict
    
@torch.no_grad()
def valid_one_epoch(cfg, model, dataloader, criterion, device, epoch, stat_dict, run_log_wandb):
    model.eval()
    
    #avg_ssim = AverageMeter()
    count = 0
    test_log = ""
    name = 'PER'
    
    acc_all = []
    for _dl in dataloader:
        name = _dl['name']
        dataloader = _dl['dataloader']
        running_metrics_val = runningScore(n_classes=cfg.train_config.nclass)
        val_loss_meter = AverageMeter()
        pbar = tqdm(dataloader, total=len(dataloader), desc='Valid: {}'.format(name))
        count = 0
        
        acc_db = []
        dirPath = "./result_image/val/{}/".format(name)
                 
        for _image_patch, _gt_patch, _idx in pbar:
            
            _image_patch, _gt_patch = _image_patch.to(device), _gt_patch.to(device)
            
            _start_pixel_err = time.time()
            
            _pred = model(_image_patch)
            
            _end_pixel_err = time.time()
            
            count += _image_patch.size(0)
            
            _imgPred = []
            if cfg.train_config.pixel_recovery:
                #pred = np.squeeze(_pred.data.max(1)[1].cpu().numpy(), axis=0)
                #gt = np.squeeze(_gt_patch.data.cpu().numpy(), axis=0)

                ### Pred data
                for b in range(_pred.shape[0]):
                    #pred = _pred.data.max(1)[1].squeeze(axis=0)
                    pred = _pred.data.max(1)[1][b]
                    #pred = _pred[b]
                    unique, counts = pred.unique(return_counts=True)
                    if len(counts) >1:
                        total_erro_pixel_Pred = counts[1]
                    else: 
                        total_erro_pixel_Pred = 1
                        
                    index = torch.where(pred==1)
                    
                    #GT date
                    gt = _gt_patch[b]
                    unique, counts = gt.unique(return_counts=True)
                    if len(counts) >1:
                        total_erro_pixel_GT = counts[1]
                    else: 
                        total_erro_pixel_Pred = 1
                    
                    acc_rec = accuracy_score(gt.cpu(), pred.cpu())
                    #acc_rec = np.round(float(total_erro_pixel_Pred/total_erro_pixel_GT),2)
                    acc_db.append(acc_rec)
                    
                    ### --- 
                    ### --- 
                    # Image Reconstruct with predicted ERROR
                    img = _image_patch[b].data.cpu().numpy().transpose(1,2,0)
                    _start_pixel_rec = time.time()
                    nH, nW = img.shape[0], img.shape[1]
                    _imgPred = np.full((nH+4, nW+4, 3), (128,128,128))
                    _imgPred[1:nH+1,1:nW+1,:] = img*255
                    
                    #index = np.where(_pred_one==1)
                    #_index = np.add(index,1)
                    
                    #for _id0, _id1 in zip(index[0,:], index[1,:]):
                    for _id0, _id1 in zip(index[0].data.cpu().numpy(), index[1].data.cpu().numpy()):
                        _id0 += 1
                        _id1 += 1
                        crop_neighbor = np.array(_imgPred[_id0-1:_id0+1,_id1-1:_id1+1,:])
                        _imgPred[_id0,_id1,:] = (median(np.sort(crop_neighbor[:,:,0].ravel())),median(np.sort(crop_neighbor[:,:,1].ravel())),median(np.sort(crop_neighbor[:,:,2].ravel())))
                
                _end_pixel_rec = time.time()
                
                if cfg.train_config.save_img_rec:
                    fname = str(name) + '_munster_' + str(int(_idx)).zfill(6) + '_000019_leftImg8bit_recover.jpg'
                    save_img(os.path.join(dirPath, str(epoch), fname), _imgPred[1:nH+1,1:nW+1,:].astype(np.uint8), color_domain='rgb')
                
                    with open(os.path.join(dirPath,"result.csv"), "a") as file:
                        file.write("{},{},{},{}, {:.4f}, {:.4f}, {:.4f} \n".format(str(fname),int(total_erro_pixel_GT.data.cpu().numpy()), int(total_erro_pixel_Pred.data.cpu().numpy()), str(acc_rec), (_end_pixel_rec-_start_pixel_err), (_end_pixel_err-_start_pixel_err),(_end_pixel_rec-_start_pixel_rec)))
                    
            if cfg.train_config.img_save_val and count < 81:  
                pred = _pred.data.max(1)[1].cpu().numpy()
                gt = _gt_patch.data.cpu().numpy()
                img = _image_patch.data.cpu().numpy()
                
                fname = str(count).zfill(4)
                
                _pred = pred[0]
                _pred = np.expand_dims(_pred,axis=0)
                _pred_one = np.squeeze(_pred,axis=0)
                
                _pred = np.concatenate((_pred,_pred,_pred), axis=0).transpose(1,2,0)
                
                _pred[_pred == 1] = 255
                
                _gt = gt[0]
                _gt = np.expand_dims(_gt,axis=0)
                _gt = np.concatenate((_gt,_gt,_gt), axis=0).transpose(1,2,0)
                
                if cfg.train_config.colors == 1:
                    _img = img[0]
                    _img = np.concatenate((_img, _img, _img), axis=0).transpose(1,2,0) * 255
                else: 
                    _img = img[0].transpose(1,2,0)*255
                
                _pred[_pred == 0] = 0
                _pred[_pred == 1] = 255
                
                _gt[_gt == 0] = 0
                _gt[_gt == 1] = 255
                
                ### ---
                ### ---
                ### Save images
                temp0 = np.concatenate((_gt, _img, _pred, _imgPred[1:nH+1,1:nW+1,:]),axis=1)
                save_img(os.path.join(dirPath, str(epoch), fname + '.jpg'), temp0.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_gt.jpg'), _gt.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_img.jpg'), _img.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_pred.jpg'), _pred.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_img_pred.jpg'), _imgPred.astype(np.uint8), color_domain='rgb')
                
            pbar.set_postfix(epoch=f'{epoch}',
                        acc=f'{acc_rec:0.4f}')
        
        score, class_iou = running_metrics_val.get_scores()
        _score=[]
        
        for k, v in score.items():
           _score.append(v)
           
        print('Accuracy mean({}) = {}'.format(name, mean(acc_db)))
        acc_all.append(mean(acc_db))
        
        running_metrics_val.reset()    
        
        if cfg.train_config.wandb:
            # Log the metrics
            wandb.log({
                "val/Valid Loss ({})".format(name): val_loss_meter.avg,
                "val/Valid ACC {} - ".format(name): mean(acc_db),
                })
    
    print("ALL Accuracy:::", mean(acc_all)) 
    if cfg.train_config.wandb:
        # Log the metrics
        wandb.log({
            "val/Valid Loss ({})".format(name): val_loss_meter.avg,
            "val/Valid Acc-Mean ({})".format(name): mean(acc_all)
            })
    return mean(acc_all)

def run_training(cfg, model, optimizer, scheduler, criterion, device, num_epochs, train_loader, valid_loader, run_log_wandb):
    # To automatically log gradients
    #wandb.watch(model, log_freq=100)
    
    start = time.time()
    best_miou      = -np.inf
    best_epoch     = -1
    
    if cfg.train_config.pretrain is not None:
        print("load pretrained model: {}!".format(cfg.train_config.pretrain))
        model.load_state_dict(torch.load(cfg.train_config.pretrain))
    else:
        print("train the model from scratch!")

    print("Check whether the pretrained model is restored...")
    if cfg.train_config.resume:
        # Load checkpoint model
        checkpoint = torch.load(cfg.train_config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        cfg.train_config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        # scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")
    
    log_name = os.path.join("log.txt")
    sys.stdout = utils_sr.ExperimentLogger(log_name, sys.stdout)
    stat_dict = utils_sr.get_stat_dict()
        
    for epoch in range(cfg.train_config.start_epoch, num_epochs + 1):
        acc = 0
        print(f'Epoch {epoch}/{num_epochs}', end='')
        
        train_one_epoch(cfg, model, optimizer, scheduler, criterion=criterion,
                                       dataloader=train_loader,
                                       device=device, epoch=epoch, stat_dict=stat_dict, run_log_wandb=run_log_wandb)
        
        if (epoch % cfg.train_config.test_every) == 0:
            acc = valid_one_epoch(cfg, model, valid_loader, criterion,
                                        device=device,
                                        epoch=epoch, stat_dict=stat_dict, run_log_wandb=run_log_wandb)
        
        #epoch_PSNR = val_div2k_psnr
        if acc > best_miou:
            print("{}Valid ACC Improved at {} -> (Before:{} ---> Best:{})".format(c_,'all',best_miou,acc))
            #print("\t{}Valid SSIM Improved at {} -> ({}), Epoch: {}/{}".format(c_,'Div2k',stat_dict['Div2k']['best_ssim']['value'], epoch, num_epochs))
            sys.stdout.flush()
            
            best_miou = acc
            best_epoch        = epoch
            if cfg.train_config.wandb:
                wandb.summary["Best Acc"]    = acc
                wandb.summary["Best Epoch"]   = best_epoch

            # save stat dict
            ## save training paramters
            # stat_dict_name = os.path.join('./', 'stat_dict.yml')
            # with open(stat_dict_name, 'w') as stat_dict_file:
            #     yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        
            dirPath = "./run/{}".format(cfg.train_config.comment)
            PATH = f"best_epoch.pt"
            torch.save(model.state_dict(), os.path.join(dirPath,PATH))
            # Save a model file from the current directory
            if cfg.train_config.wandb:
                wandb.save(PATH)
            print(f"Model Saved{sr_}")

        #last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch.pt"
        torch.save(model.state_dict(), PATH)
        #print(); print()
        
        torch.cuda.empty_cache()
        gc.collect()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best ACC Score: {:.4f}".format(best_miou))
    return model

@hydra.main(config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:
    set_seed()
    
    device = None
    if cfg.train_config.gpu_id >= 0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(cfg.train_config.gpu_id))
        device = torch.device('cuda:{}'.format(cfg.train_config.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')
    #torch.set_num_threads(cfg.train_config.threads)
    
    ### For use original path.
    ### For debug...
    #currPath = os.getcwd()
    #os.chdir(currPath)
    #print(currPath)
    # org_cwd = hydra.utils.get_original_cwd()
    # print(org_cwd)
    
    dirPath = "./run/{}".format(cfg.train_config.comment)
    if not os.path.isdir(dirPath): 
        os.makedirs(dirPath)
    
    model = get_model(cfg, device)
    train_loader, valid_loader  = get_dataset(cfg)
    optimizer = get_optimizer(cfg, model)
    
    scheduler = get_scheduler(cfg, optimizer)
    criterion = get_criterion(cfg)
    print(cfg.train_config.comment)
    
    if cfg.train_config.wandb:
        run_log_wandb = wandb.init(entity="gnzrg25", project='Noise pixel detection',
                    config={k:v for k, v in dict(cfg).items() if '__' not in k},
                    anonymous=anonymous,
                    name=f"dim-{cfg.train_config.patch_size}x{cfg.train_config.patch_size}|model-{cfg.model.name}",
                    group=cfg.train_config.comment,
                    )
    else: 
        run_log_wandb = 0
    
    model = run_training(cfg, model, optimizer, scheduler, criterion=criterion,
                                device=device,
                                num_epochs=cfg.train_config.epochs,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                run_log_wandb=run_log_wandb)
    if cfg.train_config.wandb:
        run_log_wandb.finish()
        
@hydra.main(config_path="conf", config_name="config")
def validate(cfg : DictConfig) -> None:
    device = None
    if cfg.train_config.gpu_id >= 0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(cfg.train_config.gpu_id))
        device = torch.device('cuda:{}'.format(cfg.train_config.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')

    dirPath = "./run/validate/{}".format(cfg.train_config.comment)
    if not os.path.isdir(dirPath): 
        os.makedirs(dirPath)
    
    model = get_model(cfg, device)
    _, valid_loader  = get_dataset(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    criterion = get_criterion(cfg)
    print(cfg.train_config.comment)
    
    model = run_validate(cfg, model, optimizer, scheduler, criterion=criterion,
                                device=device,
                                num_epochs=cfg.train_config.epochs,
                                train_loader=None,
                                valid_loader=valid_loader, 
                                model_path= cfg.train_config.testmodelpath)
if __name__ == '__main__':
    train()
    #validate()