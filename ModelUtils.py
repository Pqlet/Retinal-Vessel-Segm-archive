from typing import Tuple, List, Callable, Iterator, Optional, Dict, Any
from collections import defaultdict
import torch.nn as nn
import torch
import numpy as np
import os

import torchvision.transforms
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import albumentations as A
from MetricsUtils import SoftDice

from monai.inferers import sliding_window_inference
import cv2
from UtilsMasks import *
from Utils import *


class UnetTrainer:
    """
    Класс, реализующий обучение модели
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 device: str,
                 flag_patches: bool,
                 metric_functions: List[Tuple[str, Callable]] = [],
                 epoch_number: int = 0,
                 lr_scheduler: Optional[Any] = None):
        self.flag_patches = flag_patches
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.metric_functions = metric_functions

        self.epoch_number = epoch_number
        self.not_saved_val_epoch = 0

    @torch.no_grad()
    def evaluate_batch(self, val_iterator: Iterator, eval_on_n_batches: int) -> Optional[Dict[str, float]]:
        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(eval_on_n_batches):
            try:
                batch = next(val_iterator)

                xs = batch['image'].to(self.device)
                ys_true = batch['mask'].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break
            if self.flag_patches:
                # For patches
                ys_pred = sliding_window_inference(
                     xs.to(self.device),
                     roi_size=(224, 224),
                     sw_batch_size=8,
                     overlap=0,
                     predictor=self.model.eval(),
                     device=self.device
                     )
                ys_pred = torchvision.transforms.CenterCrop((1232, 1624))(ys_pred)
                ys_true = torchvision.transforms.CenterCrop((1232, 1624))(ys_true)
                if self.not_saved_val_epoch != self.epoch_number:
                    for i, mask_pred in enumerate(ys_pred):
                        print("saving_image")
                        image_mask_to_save = mask2ch_to_img(mask_pred.cpu(), (3, 1232, 1624), 0.5)
                        cv2.imwrite(os.path.join('IMAGES_FROM_VAL',f'{self.epoch_number}__{i}.png'),
                                    image_mask_to_save)
                    self.not_saved_val_epoch = self.epoch_number

            else:
                # For a full image
                ys_pred = self.model.eval()(xs)
            loss = self.criterion(ys_pred, ys_true)


            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader, eval_on_n_batches: int = 1) -> Dict[str, float]:
        """
        Вычисление метрик для эпохи
        """
        metrics_sum = defaultdict(float)
        num_batches = 0

        val_iterator = iter(val_loader)

        with tqdm(total=len(val_loader.dataset) // val_loader.batch_size // eval_on_n_batches,
                  leave=None,
                  desc='Val loader') as pbar_epoch:
            while True:
                batch_metrics = self.evaluate_batch(val_iterator, eval_on_n_batches)
                if batch_metrics is None:
                    break
                for metric_name in batch_metrics:
                    metrics_sum[metric_name] += batch_metrics[metric_name]

                num_batches += 1
                pbar_epoch.update(1)

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit_batch(self, train_iterator: Iterator, update_every_n_batches: int) -> Optional[Dict[str, float]]:
        """
        Тренировка модели на одном батче
        """

        self.optimizer.zero_grad()

        predictions = []
        targets = []

        losses = []

        for real_batch_number in (range(update_every_n_batches)):
            try:
                batch = next(train_iterator)
                xs = batch['image'].to(self.device)
                ys_true = batch['mask'].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break
            ys_pred = self.model.train()(xs)
            loss = self.criterion(ys_pred, ys_true)

            (loss / update_every_n_batches).backward()

            losses.append(loss.item())

            predictions.append(ys_pred.detach().cpu())
            targets.append(ys_true.detach().cpu())

        self.optimizer.step()

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        return metrics

    def fit_epoch(self, train_loader, update_every_n_batches: int = 1) -> Dict[str, float]:
        """
        Одна эпоха тренировки модели
        """

        metrics_sum = defaultdict(float)
        num_batches = 0

        train_iterator = iter(train_loader)
        with tqdm(total=len(train_loader.dataset) // train_loader.batch_size // update_every_n_batches + 1,
                  leave=None,
                  desc='Train loader') as pbar_epoch:
            while True:

                batch_metrics = self.fit_batch(train_iterator, update_every_n_batches)

                if batch_metrics is None:
                    break

                for metric_name in batch_metrics:
                    metrics_sum[metric_name] += batch_metrics[metric_name]

                num_batches += 1
                pbar_epoch.update(1)
        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit(self, train_loader, num_epochs: int,
            val_loader=None, update_every_n_batches: int = 1,
            train_folder=None,
            early_stopping_epochs=None
            ):
        """
        Метод, тренирующий модель и вычисляющий метрики для каждой эпохи
        Returns
            Summary, best_model
        """
        assert early_stopping_epochs > 0, 'early_stopping_epochs should be > 0'
        if self.flag_patches is None:
            train_folder = os.path.join('train_history', f'spare_train_folder_{get_curr_time()}')
            os.mkdir(train_folder)
        summary = defaultdict(list)
        best_val_metric = 0
        impatience = 0
        best_model_path = None

        def save_metrics(metrics: Dict[str, float], postfix: str = '') -> None:
            # Сохранение метрик в summary
            nonlocal summary, self

            for metric in metrics:
                metric_name, metric_value = f'{metric}{postfix}', metrics[metric]

                summary[metric_name].append(metric_value)

        for _ in tqdm(range(num_epochs - self.epoch_number),
                      initial=self.epoch_number,
                      total=num_epochs,
                      desc='Epochs',
                      leave=True):
            if impatience >= early_stopping_epochs:
                break
            self.epoch_number += 1
            print(f'Epoch {self.epoch_number}/{num_epochs}')
            train_metrics = self.fit_epoch(train_loader, update_every_n_batches)

            with torch.no_grad():
                save_metrics(train_metrics, postfix='_train')

                if val_loader is not None:
                    test_metrics = self.evaluate(val_loader)
                    save_metrics(test_metrics, postfix='_test')
            print('\nTrain metrics\n{}\nTest metrics\n{}'.format(train_metrics, test_metrics))
            with open(os.path.join(train_folder, 'summary.txt'), 'w') as f:
                print(summary, file=f)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if test_metrics['exp_dice'] > best_val_metric:
                best_val_metric = test_metrics['exp_dice']
                best_model_path = os.path.join(train_folder, f'model_checkpoint_best.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f'Checkpoint was made. Best metric = {best_val_metric}')
                impatience = 0
            else:
                impatience += 1

        try:
            last_model_path = os.path.join(train_folder, f'model_checkpoint_last.pth')
            torch.save(self.model.state_dict(), last_model_path)
        finally:
            print(f'Saved the last model')

        summary = {metric: np.array(summary[metric]) for metric in summary}
        if best_model_path is not None:
            self.model.load_state_dict(torch.load(best_model_path))
        return summary, self.model


def make_criterion(criterion_name='SoftDice'):
    """
    :param criterion_name: One of ["SoftDice", "BCE"]
    :return:
    """
    if criterion_name=='SoftDice':
        soft_dice = SoftDice()
        def exp_dice(pred, target):
            return 1 - soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])
        critetrion=exp_dice
    elif criterion_name=='BCE':
        bce = nn.BCELoss()
        def bce_f(pred, target):
            return bce(torch.exp(pred[:, 1:]), target[:, 1:])
        critetrion=bce_f
    else:
        assert False, "Incorrect criterion_name"
    return critetrion