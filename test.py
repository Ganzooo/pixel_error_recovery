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

c_ = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings

warnings.filterwarnings("ignore")

# Wandb and mlflow
import wandb
import cv2
from fvcore.nn import FlopCountAnalysis as fca
from thop import profile

try:
    wandb.login(key='0b0a03cb580e75ef44b4dff7f6f16ce9cfa8a290')
    anonymous = None
except:
    anonymous = "must"
    print(
        'To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')


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
                    loss = criterion(_pred, _gt_patch)
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
                    _img = np.concatenate((_img, _img, _img), axis=0).transpose(1, 2, 0) * 255
                else:
                    _img = img[0].transpose(1, 2, 0) * 255

                _pred = pred[0]
                _pred = np.expand_dims(_pred, axis=0)
                _pred = np.concatenate((_pred, _pred, _pred), axis=0).transpose(1, 2, 0)

                _gt = gt[0]
                _gt = np.expand_dims(_gt, axis=0)
                _gt = np.concatenate((_gt, _gt, _gt), axis=0).transpose(1, 2, 0)

                _pred[_pred == 0] = 0
                _pred[_pred == 1] = 255

                _gt[_gt == 0] = 0
                _gt[_gt == 1] = 128

                temp0 = np.concatenate((_gt, _img, _pred), axis=1)
                save_img(os.path.join(dirPath, str(epoch), fname + '.jpg'), temp0.astype(np.uint8), color_domain='rgb')

            # if cfg.train_config.debug and step < 10:

    if cfg.train_config.wandb:
        # Log the metrics
        wandb.log({"train/Loss": epoch_loss.avg,
                   "train/LR": scheduler.get_last_lr()[0]})

    stat_dict['losses'].append(epoch_loss.avg)
    return epoch_loss.avg, stat_dict


@torch.no_grad()
def valid_one_epoch(cfg, model, dataloader, criterion, device, epoch, stat_dict, run_log_wandb):
    model.eval()

    # avg_ssim = AverageMeter()
    count = 0
    test_log = ""
    name = 'PER'

    acc_all = []
    psnr_all = []
    ssim_all = []
    inp = None

    total_latency = 0
    warmup_cnt = 0

    for _dl in dataloader:
        name = _dl['name']
        dataloader = _dl['dataloader']
        running_metrics_val = runningScore(n_classes=cfg.train_config.nclass)
        val_loss_meter = AverageMeter()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid: {}'.format(name))
        count = 0

        dirPath = "./result_image/val/{}/".format(name)
        if not os.path.isdir(dirPath):
            os.makedirs(dirPath)

        if not os.path.exists(os.path.join(dirPath, "result.csv")):
            with open(os.path.join(dirPath, "result.csv"), "w") as file:
                file.write("name, acc, psnr, ssim, total\n")

        acc_db = []
        psnr_db = []
        ssim_db = []

        for data in dataloader:
            _image_patch, _gt_patch, _idx, _fname, _imageOrg_patch = data
            _image_patch, _gt_patch = _image_patch.to(device), _gt_patch.to(device)
            _pred, _rec_img = model(_image_patch)
            warmup_cnt += 1

            if warmup_cnt > 100:
                break

        for step, data in pbar:
            _image_patch, _gt_patch, _idx, _fname, _imageOrg_patch = data
            _image_patch, _gt_patch = _image_patch.to(device), _gt_patch.to(device)

            _start_pixel_err = time.time()
            _pred, _rec_img = model(_image_patch)
            _end_pixel_err = time.time()
            total_latency_ = _end_pixel_err - _start_pixel_err
            total_latency += total_latency_

            count += _image_patch.size(0)

            if inp is None:
                inp = _image_patch

            _imgPred = []
            _imgRec = []
            if cfg.train_config.pixel_recovery:
                filter_size = cfg.train_config.fsize

                ### Pred data
                for b in range(_pred.shape[0]):
                    # pred = _pred.data.max(1)[1].squeeze(axis=0)
                    pred = _pred.data.max(1)[1][b]
                    # index = torch.where(pred==1)

                    gt = _gt_patch[b]
                    rec_img = _rec_img * 255

                    acc_rec = accuracy_score(gt.cpu(), pred.cpu())
                    # acc_rec = np.round(float(total_erro_pixel_Pred/total_erro_pixel_GT),2)
                    acc_db.append(acc_rec)

                    _imgRec = rec_img[b].permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                    _imgOrg = _imageOrg_patch[b].numpy()

                    psnr = psnr_calc(_imgOrg, _imgRec)
                    psnr_db.append(psnr)

                    ssim_R = ssim_calc(_imgOrg[:, :, 0], _imgRec[:, :, 0], full=True)
                    ssim_G = ssim_calc(_imgOrg[:, :, 1], _imgRec[:, :, 1], full=True)
                    ssim_B = ssim_calc(_imgOrg[:, :, 2], _imgRec[:, :, 2], full=True)
                    ssim = mean([ssim_R[0], ssim_G[0], ssim_B[0]])
                    ssim_db.append(ssim)

                fname = os.path.basename(_fname[0])

                if step % 100 == 0:
                    with open(os.path.join(dirPath, "result.csv"), "a") as file:
                        file.write(
                            "{}, {:.6f}, {:.6f}, {:.6f}, {:.6f} \n".format(str(fname),
                                                                           acc_rec,
                                                                           psnr,
                                                                           ssim,
                                                                           total_latency_))

                    save_img(os.path.join(dirPath, str(epoch) + '_rec', fname), _imgRec, color_domain='rgb')

            pbar.set_postfix(epoch=f'{epoch}', acc=f'{acc_rec:0.4f}', psnr=f'{psnr:0.2f}', ssim=f'{ssim:0.2f}')

        score, class_iou = running_metrics_val.get_scores()
        _score = []

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

    macs, params = profile(model, inputs=(inp,))
    flops = fca(model, inp)

    with open(os.path.join(dirPath, "flops.txt"), "a") as file:
        file.write(f"macs: {macs} / params: {params}\n")
        file.write(f"flops: {flops.total()}\n{flops.by_module_and_operator()}")

    with open(os.path.join(dirPath, "result.csv"), "a") as file:
        file.write(
            "{}, {:.6f}, {:.6f}, {:.6f}, {:.6f} \n".format('',
                                                           mean(acc_all),
                                                           mean(psnr_all),
                                                           mean(ssim_all),
                                                           total_latency / len(dataloader)))

    # print("FPS:{:.2f}".format(60/(end_t-start_t)))
    return mean(acc_all)


def run_validate(cfg, model, optimizer, scheduler, criterion, device, num_epochs, train_loader, valid_loader,
                 model_path):
    start = time.time()
    best_miou = -np.inf
    best_epoch = -1

    print("load test model: {}!".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    log_name = os.path.join("log.txt")
    sys.stdout = utils_sr.ExperimentLogger(log_name, sys.stdout)
    stat_dict = utils_sr.get_stat_dict()

    cfg.train_config.wandb = 0
    valid_one_epoch(cfg, model, valid_loader, criterion,
                    device=device,
                    epoch=0, stat_dict=stat_dict, run_log_wandb=None)

    end = time.time()
    time_elapsed = end - start
    # print('Validation complete in {:.0f}h {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best mIoU Score: {:.4f}".format(best_miou))
    return model


@hydra.main(config_path="conf", config_name="config_detect_test")
def validate(cfg: DictConfig) -> None:
    device = None
    if cfg.train_config.gpu_id >= 0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(cfg.train_config.gpu_id))
        device = torch.device('cuda:{}'.format(cfg.train_config.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')

    dirPath = "./run/{}".format(cfg.train_config.comment)
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)

    model = get_model(cfg, device)
    _, valid_loader = get_dataset(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    criterion = get_criterion(cfg)
    print(cfg.train_config.comment)

    model = run_validate(cfg, model, optimizer, scheduler, criterion=criterion,
                         device=device,
                         num_epochs=cfg.train_config.epochs,
                         train_loader=None,
                         valid_loader=valid_loader,
                         model_path=cfg.train_config.testmodelpath)


@hydra.main(config_path="conf", config_name="config_hybrid")
def torch_to_onnx(cfg: DictConfig) -> None:
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
        img = torch.zeros(1, 3, 128, 128).to(device)
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        model.eval()
        output_names = ['recovery', 'detect']  # if y is None else ['output']
        dynamic_axes = None

        torch.onnx.export(model, img, fileName, verbose=False, opset_version=12, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(fileName)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        # if opt.end2end and opt.max_wh is None:
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
        onnx.save(onnx_model, fileName)
        print('ONNX export success, saved as %s' % fileName)

    except Exception as e:
        print('ONNX export failure: %s' % e)


if __name__ == '__main__':
    validate()
# torch_to_onnx()
