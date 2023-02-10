import torch
import numpy as np
import cv2
from torchvision.utils import draw_segmentation_masks

def mask2ch_to_img(prediction, original_image_chw, thresh_mask):
    """
    :param prediction:
        Should be tensor with shape [c, h, w]
    :param original_image:
    :param thresh_mask:
    :return:
        numpy array
    """
    # This is binary matrices yet
    pred_mask = (torch.exp(prediction) > thresh_mask)
    # Converting binary matrices into black and white image-mask of the initial image
    mask_image = np.moveaxis(draw_segmentation_masks(torch.zeros(*original_image_chw, dtype=torch.uint8), pred_mask).cpu().numpy(), 0, -1)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, mask_image = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)

    return mask_image
