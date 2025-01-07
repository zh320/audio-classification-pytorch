"""
Create by:  zh320
Date:       2024/07/13
"""

import torch, random
from tqdm import tqdm
from torch.cuda import amp

from .base_trainer import BaseTrainer
from .loss import kd_loss_fn
from models import get_teacher_model
from utils import (get_cls_metrics, sampler_set_epoch)


class ClsTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if not config.is_testing:
            self.teacher_model = get_teacher_model(config, self.device)
            self.metrics = get_cls_metrics(config).to(self.device)

    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (inputs, labels) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            # Forward path
            with amp.autocast(enabled=config.amp_training):
                preds = self.model(inputs)
                loss = self.loss_fn(preds, labels)

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)

            # Knowledge distillation
            if config.kd_training:
                with amp.autocast(enabled=config.amp_training):
                    with torch.no_grad():
                        teacher_preds = self.teacher_model(inputs)   # Teacher predictions

                    loss_kd = kd_loss_fn(config, preds, teacher_preds.detach())
                    loss += config.kd_loss_coefficient * loss_kd

                if config.use_tb and self.main_rank:
                    self.writer.add_scalar('train/loss_kd', loss_kd.detach(), self.train_itrs)
                    self.writer.add_scalar('train/loss_total', loss.detach(), self.train_itrs)

            # Backward path
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.ema_model.update(self.model, self.train_itrs)

            if self.main_rank:
                pbar.set_description(('%s'*2) % 
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.detach():4.4g}{" "*4}|',)
                                )

        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for (inputs, labels) in pbar:
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            preds = self.ema_model.ema(inputs)
            self.metrics.update(preds.detach(), labels)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        score = self.metrics.compute()

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' + 
                                 f'\n\nBest accuracy is: {score:.4f}\n')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} Accuracy: {score:.4f}    | ' + 
                                 f'best accuracy so far: {self.best_score:.4f}\n')

            if config.use_tb and self.cur_epoch < config.total_epoch:
                self.writer.add_scalar('val/accuracy', score.cpu(), self.cur_epoch+1)

        self.metrics.reset()
        return score

    @torch.no_grad()
    def predict(self, config):
        if config.DDP:
            raise NotImplementedError('Predict mode currently does not support DDP.')

        import os, json
        save_path = os.path.join(config.save_dir, 'pred_results.json')

        self.logger.info('\nStart predicting...\n')

        self.model.eval()

        pred_results = {}
        for (inputs, img_names) in tqdm(self.test_loader):
            inputs = inputs.to(self.device, dtype=torch.float32)

            preds = self.model(inputs)

            preds_cls = torch.argmax(preds, dim=1)

            for i in range(preds.shape[0]):
                img_name = img_names[i]
                pred_cls = preds_cls[i].item()

                pred_results[img_name] = config.class_map[pred_cls]

        with open(save_path, 'w') as f:
            json.dump(pred_results, f, indent=1)

        self.logger.info(f'\nPrediction finished. Results are saved at {save_path}.\n')
