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
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import structural_similarity as ssim_calc

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

#Wandb and mlflow
import wandb
import cv2 

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
    psnr_all = []
    ssim_all = []
    start_t = time.perf_counter() 
    
    for _dl in dataloader:
        name = _dl['name']
        dataloader = _dl['dataloader']
        running_metrics_val = runningScore(n_classes=cfg.train_config.nclass)
        val_loss_meter = AverageMeter()
        pbar = tqdm(dataloader, total=len(dataloader), desc='Valid: {}'.format(name))
        count = 0
        
        acc_db = []
        psnr_db = []
        ssim_db = []
        dirPath = "/mnt/data_main/NETWORK/11_IMAGE_DENOISE/pixel_error_recovery/result_image/val/{}/".format(name)
               
        for _image_patch, _gt_patch, _idx, _fname, _imageOrg_patch in pbar:
            
            _image_patch, _gt_patch = _image_patch.to(device), _gt_patch.to(device)
            
            _start_pixel_err = time.time()
            
            _pred = model(_image_patch)
            
            _end_pixel_err = time.time()
            
            count += _image_patch.size(0)
            
            _imgPred = []
            _imgRec = []
            if cfg.train_config.pixel_recovery:
                filter_size = cfg.train_config.fsize

                ### Pred data
                for b in range(_pred.shape[0]):
                    #pred = _pred.data.max(1)[1].squeeze(axis=0)
                    pred = _pred.data.max(1)[1][b]
                    index = torch.where(pred==1)
                    
                    gt = _gt_patch[b]
                    
                    acc_rec = accuracy_score(gt.cpu(), pred.cpu())
                    #acc_rec = np.round(float(total_erro_pixel_Pred/total_erro_pixel_GT),2)
                    acc_db.append(acc_rec)
                    
                    ### --- 
                    ### --- 
                    # Image Reconstruct with predicted ERROR
                    img = _image_patch[b].data.cpu().numpy().transpose(1,2,0)
                    _start_pixel_rec = time.time()
                    nH, nW = img.shape[0], img.shape[1]
                    _imgPred = np.full((nH+(filter_size*2), nW+(filter_size*2), 3), (128,128,128))
                    _imgPred[filter_size:nH+filter_size,filter_size:nW+filter_size,:] = img*255
                    
                    for _id0, _id1 in zip(index[0].data.cpu().numpy(), index[1].data.cpu().numpy()):
                        _id0 += filter_size
                        _id1 += filter_size
                        crop_neighbor = np.array(_imgPred[_id0-filter_size:_id0+filter_size,_id1-filter_size:_id1+filter_size,:])
                        _imgPred[_id0,_id1,:] = (median(np.sort(crop_neighbor[:,:,0].ravel())),median(np.sort(crop_neighbor[:,:,1].ravel())),median(np.sort(crop_neighbor[:,:,2].ravel())))
                    _imgRec = _imgPred[filter_size:nH+filter_size,filter_size:nW+filter_size,:].astype(np.uint8)
                    
                    _imgOrg = _imageOrg_patch[b].numpy()
                    psnr = psnr_calc(_imgOrg,_imgRec)
                    
                    if _imgOrg.shape[2] == 3:
                        ssim_R = ssim_calc(_imgOrg[:,:,0],_imgRec[:,:,0], full=True)
                        ssim_G = ssim_calc(_imgOrg[:,:,1],_imgRec[:,:,1], full=True)
                        ssim_B = ssim_calc(_imgOrg[:,:,2],_imgRec[:,:,2], full=True)
                        ssim = mean([ssim_R[0], ssim_G[0], ssim_B[0]])
                    ssim_db.append(ssim)
                    psnr_db.append(psnr)
                    
                    _end_pixel_rec = time.time()

                    fname = os.path.basename(_fname[b])
                    with open(os.path.join(dirPath,"result.csv"), "a") as file:
                        file.write("{},{},{},{}, {:.4f}, {:.4f}, {:.4f} \n".format(str(fname), str(acc_rec),str(psnr), str(ssim),(_end_pixel_rec-_start_pixel_err), (_end_pixel_err-_start_pixel_err),(_end_pixel_rec-_start_pixel_rec)))
                            
                    if cfg.train_config.save_img_rec:
                        #fname = str(name) + '_munster_' + str(int(_idx[b])).zfill(6) + '_000019_leftImg8bit_recover.png'
                        save_img(os.path.join(dirPath, str(epoch)+'_rec', fname), _imgRec.astype(np.uint8), color_domain='rgb')
                    
            if cfg.train_config.img_save_val:  
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
                
                ### Save images
                temp0 = np.concatenate((_img/255., _pred, _imgRec/255.),axis=1)
                #save_img(os.path.join(dirPath, str(epoch), fname + '.jpg'), temp0.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_gt.jpg'), _gt.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_img.jpg'), _img.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_pred.jpg'), _pred.astype(np.uint8), color_domain='rgb')
                #save_img(os.path.join(dirPath, str(epoch), fname + '_img_pred.jpg'), _imgPred.astype(np.uint8), color_domain='rgb')
                
                cv2.namedWindow("Input Image & Predicted Error & Reconstructed Image",0);
                cv2.resizeWindow("Input Image & Predicted Error & Reconstructed Image", 1920, 720);
                cv2.imshow("Input Image & Predicted Error & Reconstructed Image", temp0[:,:,::-1])
                cv2.waitKey(0)
                
            pbar.set_postfix(epoch=f'{epoch}', acc=f'{acc_rec:0.4f}', psnr=f'{psnr:0.2f}', ssim=f'{ssim:0.2f}')
        cv2.destroyAllWindows()
        score, class_iou = running_metrics_val.get_scores()
        _score=[]
        
        for k, v in score.items():
           _score.append(v)
        
        print('Accuracy mean({}) = {}'.format(name, mean(acc_db)))
        acc_all.append(mean(acc_db))
        
        if cfg.train_config.pixel_recovery:
            print('PSNR mean({}) = {}'.format(name, mean(psnr_db)))    
            psnr_all.append(mean(psnr_db))
            
            print('SSIM mean({}) = {}'.format(name, mean(ssim_db)))    
            ssim_all.append(mean(ssim_db))
            
        running_metrics_val.reset()    
        
        if cfg.train_config.wandb:
            if cfg.train_config.pixel_recovery:
                                # Log the metrics
                wandb.log({
                    "val/Valid Loss ({})".format(name): val_loss_meter.avg,
                    "val/Valid ACC {} - ".format(name): mean(acc_db),
                    "val/Valid PSNR {} - ".format(name): mean(psnr_db),
                    "val/Valid SSIM {} - ".format(name): mean(ssim_db),
                    })
            else: 
                # Log the metrics
                wandb.log({
                    "val/Valid Loss ({})".format(name): val_loss_meter.avg,
                    "val/Valid ACC {} - ".format(name): mean(acc_db),
                    })
    end_t = time.perf_counter()
    print("ALL Accuracy:::", mean(acc_all)) 
    
    #print("FPS:{:.2f}".format(60/(end_t-start_t)))
    return mean(acc_all)

def run_validate(cfg, model, optimizer, scheduler, criterion, device, num_epochs, train_loader, valid_loader, model_path):
    
    start = time.time()
    best_miou      = -np.inf
    best_epoch     = -1

    print("load test model: {}!".format(model_path))
    model.load_state_dict(torch.load(model_path))
    
    log_name = os.path.join("log.txt")
    sys.stdout = utils_sr.ExperimentLogger(log_name, sys.stdout)
    stat_dict = utils_sr.get_stat_dict()
    
    cfg.train_config.wandb = 0
    valid_one_epoch(cfg, model, valid_loader, criterion,
                                    device=device,
                                    epoch=0, stat_dict=stat_dict, run_log_wandb=None)
    
    end = time.time()
    time_elapsed = end - start
    #print('Validation complete in {:.0f}h {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    #print("Best mIoU Score: {:.4f}".format(best_miou))
    return model
        
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

@hydra.main(config_path="conf", config_name="config_hybrid")
def torch_to_onnx(cfg : DictConfig) -> None:
    device = None
    if cfg.train_config.gpu_id >= 0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(cfg.train_config.gpu_id))
        device = torch.device('cuda:{}'.format(cfg.train_config.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')
    
    model = get_model(cfg, device)
    print("load test model: {}!".format(cfg.train_config.testmodelpath))
    model.load_state_dict(torch.load(cfg.train_config.testmodelpath))
    simplify = True
    # ONNX export
    try:
        import onnx
        fileName = './resnet50_hybrid.onnx'
        img = torch.zeros(1, 3, 128,128).to(device)
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        model.eval()
        output_names = ['recovery', 'detect'] #if y is None else ['output']
        dynamic_axes = None

        torch.onnx.export(model, img, fileName, verbose=False, opset_version=12, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(fileName)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        #if opt.end2end and opt.max_wh is None:
        #    for i in onnx_model.graph.output:
        #        for j in i.type.tensor_type.shape.dim:
        #            j.dim_param = str(shapes.pop(0))

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

        # # Metadata
        # d = {'stride': int(max(model.stride))}
        # for k, v in d.items():
        #     meta = onnx_model.metadata_props.add()
        #     meta.key, meta.value = k, str(v)
        # onnx.save(onnx_model, f)

        if simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(onnx_model,fileName)
        print('ONNX export success, saved as %s' % fileName)

    except Exception as e:
        print('ONNX export failure: %s' % e)
        

if __name__ == '__main__':
    #validate()
    #torch_to_onnx()
    
