from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'esc50'
        self.data_root = '/path/to/your/dataset'
        self.val_fold = 5     # For ESC50

        # Model
        self.model = 'l3net'
        self.pretrained = False

        # Training
        self.total_epoch = 200
        self.base_lr = 0.005
        self.train_bs = 8

        # Validating
        self.val_bs = 16

        # Optimizer
        self.optimizer_type = 'sgd'

        # Training setting
        self.amp_training = False
        self.use_ema = True