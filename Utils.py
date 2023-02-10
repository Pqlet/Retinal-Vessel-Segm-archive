import torch
import imgaug
import random
import os
import numpy as np
from datetime import datetime

def get_curr_time():
    time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    return time_str

def seed_everything(seed=9195):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    imgaug.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
