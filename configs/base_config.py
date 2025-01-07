import os


class BaseConfig:
    def __init__(self,):
        # Dataset
        self.dataset = None
        self.data_root = None
        self.num_class = None
        self.num_channel = 1
        self.ignore_index = 255
        self.val_fold = 5     # For ESC50
        self.sample_rate = 44100

        # Model
        self.model = None
        self.timm_model = None
        self.pretrained = True

        # Training
        self.total_epoch = 200
        self.base_lr = 0.01
        self.train_bs = 16      # For each GPU

        # Validating
        self.val_bs = 16        # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1   # Epoch interval between validation
        self.top_k = 1

        # Testing
        self.is_testing = False
        self.test_bs = 16
        self.test_data_folder = None
        self.class_map = None

        # Loss
        self.loss_type = 'ce'
        self.class_weights = None

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3
        self.step_size = 5000        # For step lr

        # Optimizer
        self.optimizer_type = 'sgd'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD

        # Monitoring
        self.save_ckpt = True
        self.save_dir = 'save'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None
        self.logger_name = 'audio_cls_trainer'

        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = None
        self.base_workers = 8
        self.random_seed = 1
        self.use_ema = False

        # Augmentation
        # TO DO

        # DDP
        self.synBN = False
        self.destroy_ddp_process = True
        self.local_rank = int(os.getenv('LOCAL_RANK', -1))
        self.main_rank = self.local_rank in [-1, 0]

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = ''
        self.teacher_model = None
        self.kd_loss_type = 'kl_div'
        self.kd_loss_coefficient = 1.
        self.kd_temperature = 4.

    def init_dependent_config(self):
        if self.load_ckpt_path is None and not self.is_testing:
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'

        num_class_hub = {'esc50':50}
        if self.dataset in num_class_hub.keys():
            if self.main_rank:
                print(f'\nOverride num_class from {self.num_class} to {num_class_hub[self.dataset]}.')
            self.num_class = num_class_hub[self.dataset]

        if self.num_class is None:
            raise ValueError(f'\nPlease give the value of `num_class` for dataset: {self.dataset}')

        if self.is_testing:
            if self.main_rank:
                print(f'\nOverride dataset from `{self.dataset}` to `test` in test mode.')
            self.dataset = 'test'

            if self.class_map is None:
                raise ValueError('In test mode, you need to provide the class map for given dataset.')

            assert isinstance(self.class_map, dict)
            assert len(self.class_map) == self.num_class, 'Class map does not match the number of class.'
