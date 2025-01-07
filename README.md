# Introduction

Simplified PyTorch implementation of audio classification, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation etc.  

# Requirements

torch == 1.8.1  
torchvision  
torchaudio  
torchmetrics == 1.2.0  
loguru  
tqdm  
timm == 0.6.12 (optional)  

# Supported models

- [L3Net](models/l3net.py) [^l3net]  
- [ResNets](models/resnet.py) [^resnet]  
- [MobileNetV2](models/mobilenetv2.py) [^mobilenetv2]  
- timm [^timm]  

This repo also supports timm[^timm] which provides hundereds of pretrained models. For example, if you want to train `mobilenetv3_small` from timm, you may change the [config file](configs/my_config.py) to  

```
config.model = 'timm'
config.timm_model = 'mobilenetv3_small_100'
```

or use [command-line arguments](configs/parser.py)  

```
python main.py --model timm --timm_model mobilenetv3_small_100
```

Details of the configurations can also be found in this [file](configs/parser.py).  

[^l3net]: [Look, Listen and Learn](https://arxiv.org/abs/1705.08168)  
[^resnet]: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
[^mobilenetv2]: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
[^timm]: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)  

# Supported datasets

- [ESC50](datasets/esc50.py) [^esc50]  
  
  If you want to test datasets from torchaudio, you may refer to this [site](https://pytorch.org/audio/0.9.0/datasets.html). Noted that this site is outdated since the version of torchaudio(0.9.1) is bounded to torch(1.8.1). If you want to test datasets from newer version of torchaudio, you need to update this codebase to be compatible with newer torch. You can also download the audio files and build your own dataset following the style of `ESC50` dataset if you don't want to update the codebase.  

[^esc50]: [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50)  

# Knowledge Distillation

Currently only support the original knowledge distillation method proposed by Geoffrey Hinton.[^kd]  

[^kd]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  

# How to use

## DDP training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## DP training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

# Performances

## ESC50

| Model       | pretrained         | fold-1 acc(%)	| fold-2 acc(%) | fold-3 acc(%) | fold-4 acc(%) | fold-5 acc(%) |  paper acc(%) | Mean Accuracy(%)	|
| ----------- |:------------------:|:--------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-----------------:|
| L3Net		  |                    | 81.25			| 80.75			| 78.50			| 82.50			| 81.50			| 79.3			| 80.90				|
| ResNet18    |                    | 73.25			| 74.75			| 74.00			| 75.25			| 73.25			| n.a.			| 74.10				|
| ResNet18    | :white_check_mark: | 85.50			| 85.50			| 86.50			| 88.00			| 84.75			| n.a.			| 86.05				|
| MobileNetv2 |                    | 76.25			| 78.00			| 74.00			| 77.75			| 69.75			| n.a.			| 75.15				|
| MobileNetv2 | :white_check_mark: | 90.00			| 87.25			| 87.75			| 88.75			| 88.50			| n.a.			| 88.45				|

# References
