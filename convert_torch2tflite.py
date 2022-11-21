# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# This code was checked to work with the following library versions:
#
# ONNX-TensorFlow:  1.7.0   [pip install onnx-tf==1.7.0]
# ONNX:             1.8.0   [pip install onnx==1.8.0]
# TensorFlow:       2.4.0   [pip install tensorflow==2.4.0]
# PyTorch:          1.7.1   [pip install ]
#
# More information about ONNX-TensorFlow: https://github.com/onnx/onnx-tensorflow

import torch.nn as nn
import torch
import os

# DO NOT COMMENT THIS LINE (IT IS DISABLING GPU)!
# WHEN COMMENTED, THE RESULTING TF MODEL WILL HAVE INCORRECT LAYER FORMAT
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from source.model.model import get_model
from source.model.ecbsr import ECBSR
from source.model.plainsr import PlainSR
from source.model.tf.plainsr import plainsr_tf
from torch.utils.data import DataLoader
import math
import argparse, yaml
import source.utils.utils_sr as utils_sr
import os
from tqdm import tqdm
import tensorflow.keras.layers as TF_Layers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
from tensorflow.python.keras import backend as tf_keras_backend

import hydra
from omegaconf import DictConfig

import onnx
from onnx_tf.backend import prepare

import numpy as np
import tensorflow as tf
import imageio

class DIV2K(tf.keras.utils.Sequence):
    def __init__(self, data_root, scale_factor=3, batch_size=1, patch_size=0, type='valid', img_size=[360, 640, 3]):
        self.data_root = data_root
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.type = type
        self.min_img_size = img_size
        
        if self.type == 'train':
            self.image_ids = range(1, 801)
        elif self.type == 'valid':
            self.image_ids = range(801, 900)
        else: 
            self.image_ids = range(901, 1000)

    def __getitem__(self, idx):
        return self._get_image_pair(self.image_ids[idx])

    def __len__(self):
        if self.patch_size > 0:
            length = int(np.ceil(len(self.image_ids) / self.batch_size))
        else:
            length = 1
        return length

    def _get_image_pair(self, id):
        image_id = f'{id:04}'
        
        lr_file = f'{self.data_root}/DIV2K_{self.type}_LR_bicubic/X{self.scale_factor}/{image_id}x{self.scale_factor}.png'
        lr_image = imageio.imread(lr_file, pilmode="RGB")
        #lr_image = lr_image / 255.
        
        hr_image = np.zeros([lr_image.shape[0]*self.scale_factor,lr_image.shape[1]*self.scale_factor], dtype=np.float32)
        #hr_image = hr_image / 255.
        
        #If lr_image shape less than [360, 640, 3]
        lr_image = self._get_padding(lr_image)    
        return np.float32(lr_image), np.float32(hr_image)

    def _get_padding(self, lr_image):
        nW = lr_image.shape[0]
        nH = lr_image.shape[1]
        nC = lr_image.shape[2]
        
        if self.min_img_size[0] > lr_image.shape[0] and self.min_img_size[1] > lr_image.shape[1]:
            padded_lr_image = np.zeros([self.min_img_size[0], self.min_img_size[1], nC])
            padded_lr_image[0:nW, 0:nW,:] = lr_image
            return padded_lr_image
        elif self.min_img_size[0] > lr_image.shape[0]: 
            padded_lr_image = np.zeros([self.min_img_size[0], nH, nC]) 
            return padded_lr_image
        elif self.min_img_size[1] > lr_image.shape[1]:
            padded_lr_image = np.zeros([nW, self.min_img_size[1], nC])
            return padded_lr_image
        else: 
            return lr_image 

def evaluate(model_file, data_path, image_index=0, evaluation_data='train'):
    
    def calc_psnr(y, y_target):
        mse = np.mean((y - y_target) ** 2)
        if mse == 0:
            return 100
        return 20. * math.log10( 1. / math.sqrt(mse))

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']
    div2k = DIV2K(data_path, patch_size=0, type=evaluation_data)
    # Get lr, hr image pair
    lr, hr = div2k[image_index]
    # Check if image size can be used for inference
    if lr.shape[0] < input_shape[1] or lr.shape[1] < input_shape[2]:
        print(f'Eval image {image_index} has invalid dimensions. Expecting h >= {input_shape[1]} and w >= {input_shape[2]}.')
        return None
    # Crop lr, hr images to match fixed shapes of the tensorflow lite model
    lr = lr[:input_shape[1], :input_shape[2]]
    lr = np.expand_dims(lr, 0)
    lr = np.expand_dims(lr, -1)
    interpreter.set_tensor(input_details[0]['index'], lr)
    interpreter.invoke()
    sr = interpreter.get_tensor(output_details[0]['index']).squeeze()
    hr = hr[:output_shape[1], :output_shape[2]]
    if evaluation_data == 'train':
        return np.clip(np.round(sr * 255.), 0, 255).astype(np.uint8), np.clip(np.round(hr * 255.), 0, 255).astype(np.uint8), calc_psnr(sr, hr)
    else: 
        return np.clip(np.round(sr * 255.), 0, 255).astype(np.uint8)

@hydra.main(config_path="conf", config_name="config")
def quantize(cfg : DictConfig) -> None:
    def representative_dataset_gen():
        div2k = DIV2K(cfg.tflite_converter.div2k_val, scale_factor=cfg.train_config.scale, patch_size=0, type='valid')
        for i in range(99):
            x, _ = div2k[i]
            # Skip images that are not witin input h,w boundaries
            if x.shape[0] >= cfg.tflite_converter.nH and x.shape[1] >= cfg.tflite_converter.nW:
                # crop to input shape starting for top left corner of image
                x = x[:cfg.tflite_converter.nH, :cfg.tflite_converter.nW, :]
                x = np.expand_dims(x, 0)
                #x = np.expand_dims(x, -1)
                yield [x]
                
    # Load trained SavedModel
    model = tf.saved_model.load(cfg.tflite_converter.tf_output_folder)
    # Setup fixed input shape
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_shape = [1, cfg.tflite_converter.nH, cfg.tflite_converter.nW, cfg.tflite_converter.nC]
    concrete_func.inputs[0].set_shape(input_shape)
    # Get tf.lite converter instance
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Use full integer operations in quantized model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    
    # Set input and output dtypes to UINT8 (uncomment the following two lines to generate an integer only model)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Provide representative dataset for quantization calibration
    converter.representative_dataset = representative_dataset_gen
    # Convert to 8-bit TensorFlow Lite model
    
    tflite_model = converter.convert()
    open('{}/model_int8.tflite'.format(cfg.tflite_converter.tl_output_folder), 'wb').write(tflite_model)
   
@hydra.main(config_path="conf", config_name="config")
def convert_onnx2tflite(cfg : DictConfig) -> None:
    if not os.path.exists(cfg.tflite_converter.tl_output_folder):
        os.makedirs(cfg.tflite_converter.tl_output_folder)
    if not os.path.exists(cfg.tflite_converter.tf_output_folder):
        os.makedirs(cfg.tflite_converter.tf_output_folder)
    
    # Converting model to ONNX
    # Creating / loading pre-trained PyNET model
    
    device = torch.device('cpu')
    ## definitions of model, loss, and optimizer
    model = get_model(cfg, device)
    #model = ECBSR(module_nums=cfg.model.m_ecbsr, channel_nums=cfg.model.c_ecbsr, with_idt=cfg.model.idt_ecbsr, 
    #                    act_type=cfg.model.act_type, scale=cfg.train_config.scale, colors=cfg.train_config.colors).to(device)
    
    
    if cfg.tflite_converter.pretrain is not None:
        print("load pretrained model: {}!".format(cfg.tflite_converter.pretrain))
        model.load_state_dict(torch.load(cfg.tflite_converter.pretrain, map_location=device))
    else:
        raise ValueError('the pretrain path is invalud!')
    
    sample_input = torch.randn(cfg.tflite_converter.nB, cfg.tflite_converter.nC, cfg.tflite_converter.nH, cfg.tflite_converter.nW)
    input_nodes = ['input']
    output_nodes = ['output']

    #torch.onnx.export(model_ecbsr, sample_input, "model.onnx", export_params=True, input_names=input_nodes, output_names=output_nodes)
    onnx_file_name = os.path.join(cfg.tflite_converter.tf_output_folder,"{}.onnx".format(cfg.model.name))
    
    if cfg.model.name == 'ecbsr':
        model_plain = PlainSR(module_nums=cfg.model.m_ecbsr, channel_nums=cfg.model.c_ecbsr, act_type=cfg.model.act_type, 
                        scale=cfg.train_config.scale, colors=cfg.train_config.colors).to(device)
        ## copy weights from ecbsr to plainsr
        depth = len(model.backbone)
        for d in range(depth):
            module = model.backbone[d]
            act_type = module.act_type
            RK, RB = module.rep_params()
            model_plain.backbone[d].conv3x3.weight.data = RK
            model_plain.backbone[d].conv3x3.bias.data = RB

            if act_type == 'relu':     pass
            elif act_type == 'linear': pass
            elif act_type == 'prelu':  model_plain.backbone[d].act.weight.data = module.act.weight.data
            else: raise ValueError('invalid type of activation!')
        model_plain.eval()
        
        torch.onnx.export(
            model_plain, 
            sample_input, 
            onnx_file_name,
            input_names=input_nodes,
            output_names=output_nodes, 
            export_params=True,
            do_constant_folding=True,  
            opset_version=12
        )
    else:         
        model.eval()

        torch.onnx.export(
            model, 
            sample_input, 
            onnx_file_name,
            input_names=input_nodes,
            output_names=output_nodes, 
            export_params=True,
            do_constant_folding=True,  
            opset_version=12
        )
            
    # Converting model to Tensorflow

    onnx_model = onnx.load(onnx_file_name)
    output = prepare(onnx_model)
    #output.export_graph("tf_model/")
    output.export_graph(cfg.tflite_converter.tf_output_folder)

    # Exporting the resulting model to TFLite
    if cfg.tflite_converter.tflite_ops:
        #converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
        converter = tf.lite.TFLiteConverter.from_saved_model(cfg.tflite_converter.tf_output_folder)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TF ops
        ]
        tflite_model = converter.convert()
        
        open(os.path.join(cfg.tflite_converter.tl_output_folder,"model_fp32.tflite"), "wb").write(tflite_model)
    else: 
        converter = tf.lite.TFLiteConverter.from_saved_model(cfg.tflite_converter.tf_output_folder)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TF ops
        ]
        tflite_model = converter.convert()

        open(os.path.join(cfg.tflite_converter.tl_output_folder,"model_fp32.tflite"), "wb").write(tflite_model)

@hydra.main(config_path="conf", config_name="config")
def convert_torch2tf2tflite(cfg : DictConfig) -> None:
    if not os.path.exists(cfg.tflite_converter.tf_output_folder):
        os.makedirs(cfg.tflite_converter.tf_output_folder)
    if not os.path.exists(cfg.tflite_converter.tl_output_folder):
        os.makedirs(cfg.tflite_converter.tl_output_folder)
    
    # Converting model to ONNX
    # Creating / loading pre-trained PyNET model
    
    device = torch.device('cpu')
    ## definitions of model, loss, and optimizer
    #model_ecbsr = get_model(cfg, device)
    model_ecbsr = ECBSR(module_nums=cfg.model.m_ecbsr, channel_nums=cfg.model.c_ecbsr, with_idt=cfg.model.idt_ecbsr, 
                        act_type=cfg.model.act_type, scale=cfg.train_config.scale, colors=cfg.train_config.colors).to(device)
    model_plain = PlainSR(module_nums=cfg.model.m_ecbsr, channel_nums=cfg.model.c_ecbsr, act_type=cfg.model.act_type, 
                        scale=cfg.train_config.scale, colors=cfg.train_config.colors).to(device)
    model_plain_tf = plainsr_tf(cfg.model.m_ecbsr, cfg.model.c_ecbsr, cfg.model.act_type, 
                        cfg.train_config.scale, cfg.tflite_converter.nC, cfg.tflite_converter.nH, cfg.tflite_converter.nW)
    if cfg.tflite_converter.pretrain is not None:
        print("load pretrained model: {}!".format(cfg.tflite_converter.pretrain))
        model_ecbsr.load_state_dict(torch.load(cfg.tflite_converter.pretrain, map_location=device))
    else:
        raise ValueError('the pretrain path is invalud!')
    
    ## copy weights from ecbsr to plainsr
    depth  = len(model_plain.backbone)
    tf_idx = 0
    for d in range(depth):
        tf_idx += 1
        module = model_plain.backbone[d]
        act_type = module.act_type
        ## update weights of conv3x3
        K, B = module.conv3x3.weight, module.conv3x3.bias
        K, B = K.detach().numpy(), B.detach().numpy()
        RK_tf, RB_tf = K.transpose([2, 3, 1, 0]), B
        wgt_tf = [RK_tf, RB_tf]
        model_plain_tf.layers[tf_idx].set_weights(wgt_tf)
        ## update weights of activation
        if act_type == 'linear':
            pass
        elif act_type == 'relu':
            tf_idx += 1
        elif act_type == 'prelu':
            tf_idx += 1
            slope = module.act.weight.data
            slope = slope.view((1,1,-1))
            slope = slope.detach().numpy()
            slope_tf = slope
            wgt_tf = [slope_tf]
            model_plain_tf.layers[tf_idx].set_weights(wgt_tf)
        else:
            raise ValueError('invalid type of activation!')
    
    # save checkpoints
    model_plain_tf.save(cfg.tflite_converter.tf_output_folder, overwrite=True, include_optimizer=False, save_format='tf')        
    # # Load trained SavedModel
    model = tf.saved_model.load(cfg.tflite_converter.tf_output_folder)
    # Setup fixed input shape
    input_shape = [1, cfg.tflite_converter.nH, cfg.tflite_converter.nW, cfg.tflite_converter.nC]
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(input_shape)
    # Get tf.lite converter instance
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Use full integer operations in quantized model
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]  
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    # Exporting the resulting model to TFLite
    if cfg.tflite_converter.tflite_ops:
        #converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
        #converter = tf.lite.TFLiteConverter.from_saved_model(cfg.tflite_converter.output_folder)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TF ops
        ]
        tflite_model = converter.convert()
        
        open('{}/model_fp32.tflite'.format(cfg.tflite_converter.tl_output_folder), 'wb').write(tflite_model) 
    else: 
        #converter = tf.lite.TFLiteConverter.from_saved_model(cfg.tflite_converter.output_folder)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TF ops
        ]
        tflite_model = converter.convert()

        open('{}/model_fp32.tflite'.format(cfg.tflite_converter.tl_output_folder), 'wb').write(tflite_model) 
        
if __name__ == '__main__':
    convert_onnx2tflite()
    #convert_torch2tf2tflite()
    #quantize()