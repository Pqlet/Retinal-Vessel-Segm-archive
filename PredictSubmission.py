import torch
from monai.inferers import sliding_window_inference
import albumentations as A
import os, shutil, cv2
import numpy as np
from tqdm import tqdm
from functools import partial
from UtilsMasks import *
from Utils import *
tqdm = partial(tqdm, position=0, leave=True)

def make_submission_zip(
        model,
        flag_patches,
        subm_dataset,
        masks_to_do_list,
        original_size,
        subm_folder,
        subm_time=None,
        eval_device='cuda',
        missing_images=None,
        thresholds = (0.5,0.4,0.6,0.7)
    ):
    model.eval()
    if subm_time is None:
        subm_time=get_curr_time()
    predictions = {}
    for sample, filepath in tqdm(subm_dataset):
        # Check whether the filepath is one of those inside sample_solution.zip
        filename = filepath.split("\\")[-1]
        if filename in masks_to_do_list:
            image = sample['image']
            with torch.no_grad():
                if flag_patches:
                    # For patches 224x224
                    prediction = sliding_window_inference(
                        image.unsqueeze(dim=0).to(eval_device),
                        roi_size=(224, 224),
                        sw_batch_size=32,
                        overlap=0,
                        predictor=model.to(eval_device),
                        device=eval_device,
                    ).to('cpu')
                else:
                    # For full image
                    prediction = model.to(eval_device)(image.to(eval_device).unsqueeze(dim=0)).to('cpu')
            predictions[filename] = prediction

    for thresh_mask in thresholds:  # SAVING MASKS
        print('Threshold =', thresh_mask)
        submission_folder = os.path.join(subm_folder,f'{thresh_mask}thr__{subm_time}/')
        submission_img_folder = os.path.join(submission_folder, "img_folder")
        if not os.path.isdir(submission_folder):
            os.mkdir(submission_folder)
        if not os.path.isdir(submission_img_folder):
            os.mkdir(submission_img_folder)
        missing_images_paths = [os.path.join("submissions/sample_solution_copy", f"{num}.png") for num in
                                missing_images] if missing_images is not None else []
        for filepath in missing_images_paths:
            shutil.copy2(filepath, submission_img_folder)

        for sample, filepath in tqdm(subm_dataset):
            filename = filepath.split("\\")[-1]
            if filename in masks_to_do_list:
                # image = sample['image']
                prediction = predictions[filename][0]
                if flag_patches:
                    mask_image = mask2ch_to_img(prediction, (3, 1344, 1792), thresh_mask)
                    mask_image = A.center_crop(mask_image, *original_size)
                else:
                    mask_image = mask2ch_to_img(prediction, (3, 1248, 1632), thresh_mask)
                    mask_image = A.resize(mask_image, *original_size)
                # # Resizing for now
                # mask_image = A.Resize(*original_size)(image=mask_image)['image']
                # Saving the test image in a new folder with the original image's name
                filename_mask = filename
                cv2.imwrite(os.path.join(submission_img_folder, filename_mask), mask_image, [cv2.IMWRITE_PNG_BILEVEL, 1])

        # Making submission .zip file
        shutil.make_archive(base_name=submission_folder,
                            format='zip',
                            root_dir=submission_img_folder)