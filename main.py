import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
from os import listdir
import pandas as pd
import numpy as np
import glob

from tqdm import tqdm
from functools import partial
import warnings

import cv2
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import gc
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

tqdm = partial(tqdm, position=0, leave=True)

import random
from Utils import *
from EyeDatasetUtils import EyeDataset, DatasetPart
from ModelUtils import *
from MetricsUtils import *
from PlotUtils import *
from PredictSubmission import *

def seed_worker(worker_id):
    seed_everything()
############################################################################

def main(start_time,
         FLAG_PATCHES,
         model,
         update_every_n_batches,
         train_aug_list,
         use_train_aug,
         criterion,
         optimizer,
         scheduler,
         train_files_folder,
         original_size,
         patches_size,
         patches_size_submit,
         non_patches_size,
         non_patches_size_submit,
         fix_polygons,
         BATCH_SIZE,
         n_epochs,
         early_stopping_epochs,
         test_size,
         duplicate_dataset_int_times,
         loader_num_workers,
         device='cuda',
         SEED=9195,
         inference_model_path=None,
         training_checkpoint_path=None,
         ):

    print(start_time)

    if inference_model_path is None:
        train_model_run = True
    else:
        train_model_run = False
    eval_device = device

    if FLAG_PATCHES:
        train_size = patches_size
    else:
        train_size = non_patches_size

    train_list = [A.RandomCrop(*train_size, always_apply=True)] if FLAG_PATCHES else [
        A.LongestMaxSize(train_size[0], interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(*train_size),
        # A.ToGray(),
    ]
    if not use_train_aug:
        train_list = train_list + [ToTensorV2(transpose_mask=True)]

    eval_list = [A.PadIfNeeded(*patches_size_submit), ToTensorV2(transpose_mask=True)] if FLAG_PATCHES else [
        A.LongestMaxSize(train_size[0], interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(*train_size),
        # Validation on bigger than original like in submission
        # A.Resize(*subm_size),
        # A.ToGray(),
        ToTensorV2(transpose_mask=True),
        ]
    # Transformations for inference
    subm_list = [A.PadIfNeeded(*patches_size_submit), ToTensorV2()] if FLAG_PATCHES else [
        # A.PadIfNeeded(*non_patches_size_submit),
        A.Resize(*non_patches_size_submit),
        # A.ToGray(),
        ToTensorV2()
        ]
    transforms = {'train': A.Compose(train_list), 'test': A.Compose(eval_list), 'subm': A.Compose(subm_list)}
    augmentations = A.Compose(train_aug_list) if use_train_aug else None


    if train_model_run:
        train_files_folder = train_files_folder
        dataset = EyeDataset(train_files_folder, fix_polygons=fix_polygons)
        # Проверим состояние загруженного датасета
        for msg in dataset.make_report():
            print(msg)
        print("Обучающей выборки ", len(listdir(train_files_folder)) // 2)
        print("Тестовой выборки ", len(listdir("eye_test/")))

        test_size = test_size
        train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=SEED)

        print(f"Разбиение на train/test : {len(train_indices)}/{len(test_indices)}")
        # Разбиваем объект датасета на тренировачный и валидационный
        train_dataset = DatasetPart(dataset, train_indices,
                                    transform=transforms['train'],
                                    duplicate_dataset_int_times=duplicate_dataset_int_times,
                                    aug=augmentations)
        print(f"Train после агументаций : {len(train_dataset)}")

        valid_dataset = DatasetPart(dataset, test_indices, transform=transforms['test'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=loader_num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last = FLAG_PATCHES,
            worker_init_fn=seed_worker
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            num_workers=loader_num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker
        )

        train_time = start_time
        train_folder = f'train_history/{train_time}'
        if not os.path.isdir(train_folder):
            os.mkdir(train_folder)

        # Start TRAINING from checkpoint
        if training_checkpoint_path is not None:
            try:
                model_checkpoint_path = training_checkpoint_path
                model.load_state_dict(torch.load(model_checkpoint_path))
            finally:
                print("Checkpoint loaded")

        # Initializing loss function
        criterion = criterion
        optimizer = optimizer
        scheduler = scheduler
        trainer = UnetTrainer(model, optimizer, criterion, 'cuda',
                              flag_patches=FLAG_PATCHES,
                              metric_functions=make_metrics(),
                              lr_scheduler=scheduler,
                              )
        with open(os.path.join(train_folder, 'run_params.txt'), 'w') as f:
            print(
                f"FLAG_PATCHES:{FLAG_PATCHES}\n{'-'*50}\n"
                f"train_size:{train_size}\n{'-'*50}\n"
                f"fix_polygons:{fix_polygons}\n{'-'*50}\n"
                f"encoder_name\n{model.name}\n{'-'*50}\n"
                f"n_epochs:{n_epochs}\n{'-'*50}\n"
                f"criterion:{criterion.__name__}\n{'-'*50}\n"
                f"optimizer:{optimizer}\n{'-'*50}\n"
                f"scheduler_name:{scheduler}\n{'-'*50}\n"
                f"scheduler_params:{scheduler.__dict__}\n{'-'*50}\n"
                f"batch_size:{BATCH_SIZE}\n{'-'*50}\n"
                f"update_every_n_batches:{update_every_n_batches}\n{'-'*50}\n"
                f"early_stopping_epochs:{early_stopping_epochs}\n{'-'*50}\n"
                f"test_size_split:{test_size}\n{'-'*50}\n"
                f"train_loader:{train_loader.__dict__}\n{'-'*50}\n"
                f"valid_loader:{valid_loader.__dict__}\n{'-'*50}\n"
                f"SEED:{SEED}\n{'-'*50}\n"
                f"inference_model_path:{inference_model_path}\n{'-'*50}\n"
                f"transforms:{transforms}\n{'-'*50}\n"
                f"duplicate_dataset_int_times:{duplicate_dataset_int_times}\n{'-' * 50}\n"
                f"use_train_aug:{use_train_aug}\n{'-' * 50}\n"
                f"augmentations:{augmentations}\n{'-'*50}\n"
                f"MODEL\n{model}\n{model.segmentation_head,}\n{'-' * 50}\n",
            file=f
                )
        summary, model = trainer.fit(
            train_loader,
            num_epochs=n_epochs,
            val_loader=valid_loader,
            update_every_n_batches=update_every_n_batches,
            train_folder=train_folder,
            early_stopping_epochs=early_stopping_epochs
        )
        # ## Посмотрим метрики обученной модели на валидационном датасете
        print(summary)
        with open(os.path.join(train_folder, 'summary.txt'), 'w') as f:
            print(summary, file=f)
        torch.save(model.state_dict(), f"saved_models/Model__{train_time}.pth")
        # Функция потерь
        plot_history(summary['loss_train'], summary['loss_test'], save_folder=train_folder)

        # Сохраним несколько картинок предсказаний
        gc.collect()
        fig, axs = plt.subplots(2, 4, figsize=(16,8))
        fig.suptitle(f'Предскзаания модели {" "*105} Эталонная разметка', fontsize=14)
        model.eval()
        for i, sample in zip(range(4), valid_dataset):
            image = sample['image']
            true_mask = sample['mask'].to(eval_device)

            with torch.no_grad():
                if FLAG_PATCHES:
                    prediction = sliding_window_inference(
                     image.unsqueeze(dim=0).to(eval_device),
                     roi_size=(224, 224),
                     sw_batch_size=1,
                     overlap=0,
                     predictor=model,
                     device=eval_device
                     )
                else:
                    prediction = model.to(eval_device).eval()(image.to(eval_device).unsqueeze(dim=0))


            image = (image.cpu() * 255).type(torch.uint8)
            pred_ask = (torch.exp(prediction[0]) > 0.5).to('cpu')

            image_with_mask = draw_segmentation_masks(image, pred_ask)
            image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
            axs[i // 2, (i % 2)].imshow(image_with_mask)
            axs[i // 2, (i % 2)].axis('off')
            image_with_mask = draw_segmentation_masks(image, true_mask.type(torch.bool))
            image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
            axs[i // 2, (i % 2)+2].imshow(image_with_mask)
            axs[i // 2, (i % 2)+2].axis('off')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(f'{train_folder}/mask_preds.png')

        torch.cuda.empty_cache()
        gc.collect()

    """
    Predicting mask for submission files 
    """
    if not train_model_run:
        model.load_state_dict(torch.load(inference_model_path))
    subm_dataset = EyeDataset("eye_test/", return_filepath=True, is_test_dataset=True, transform=transforms['subm'])
    subm_time = get_curr_time() if not train_model_run else train_time
    submission_folder = f'submissions/{subm_time}'
    if not os.path.isdir(submission_folder):
        os.mkdir(submission_folder)
    masks_to_do_list = os.listdir('eye_test')  # Images to predict masks for from sample_solution
    thresholds = (0.5,)

    with open(os.path.join(submission_folder, 'run_params.txt'), 'w') as f:
        print(
            f"train_size:{train_size}\n{'-' * 50}\n"
            f"FLAG_PATCHES:{FLAG_PATCHES}\n{'-'*50}\n"
            f"MODEL\n{model.name}\n{model.segmentation_head}\n{'-'*50}\n"
            f"SEED:{SEED}\n{'-' * 50}\n"
            f"inference_model_path:{inference_model_path}\n{'-' * 50}\n"
            f"thresholds:{thresholds}\n{'-' * 50}\n",
            file=f
        )

    make_submission_zip(
        model=model,
        flag_patches=FLAG_PATCHES,
        subm_dataset=subm_dataset,
        masks_to_do_list=masks_to_do_list,
        original_size=original_size,
        subm_folder=submission_folder,
        subm_time=subm_time,
        eval_device=eval_device,
        missing_images=None,
        thresholds=thresholds
    )

    print('\n','='*40,'\n','='*17,'That\'s all folks\n','='*17,'\n','='*40)
