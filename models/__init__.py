from .resnet import ResNet
from .mobilenetv2 import Mobilenetv2
from .l3net import L3Net


def get_model(config):
    model_hub = {'l3net':L3Net, 'mobilenet_v2':Mobilenetv2}

    if config.model in model_hub.keys():
        model = model_hub[config.model](num_class=config.num_class, num_channel=config.num_channel, pretrained=config.pretrained)

    elif 'resnet' in config.model:
        model = ResNet(config.num_class, config.model, config.num_channel, config.pretrained)

    elif config.model == 'timm':
        assert config.timm_model is not None, 'You need to choose a timm model.'

        from timm import create_model
        model = create_model(config.timm_model, pretrained=config.pretrained, in_chans=config.num_channel, num_classes=config.num_class)

    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model


def get_teacher_model(config, device):
    if config.kd_training:
        import os, torch
        if not os.path.isfile(config.teacher_ckpt):
            raise ValueError(f'Could not find teacher checkpoint at path {config.teacher_ckpt}.')        

        if 'resnet' in config.teacher_model:
            model = ResNet(config.num_class, config.teacher_model, config.num_channel, pretrained=False)

        elif config.teacher_model == 'mobilenet_v2':
            model = Mobilenetv2(config.num_class, config.num_channel, pretrained=False)

        else:
            raise NotImplementedError

        teacher_ckpt = torch.load(config.teacher_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(teacher_ckpt['state_dict'])
        del teacher_ckpt

        model = model.to(device)    
        model.eval()
    else:
        model = None

    return model