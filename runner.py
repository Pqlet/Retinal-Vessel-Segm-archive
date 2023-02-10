import torch.optim.lr_scheduler

from main import *
from Utils import seed_everything
import pickle
seed_everything()

print(torch.__version__)

if __name__ == "__main__":

    start_time = get_curr_time()

    train_files_folder = "eye_train"
    test_size = 0.2
    train_aug_list = [
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=0.5),
        # Non-rigid transformations
        A.OneOf([
            A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.2, distort_limit=2, shift_limit=0.5)
        ], p=0.2),
        # Colors
        # A.RandomBrightnessContrast(p=0.4),
        # Blur
        A.OneOf([
            A.Blur(p=0.2, ),
            A.GaussianBlur(p=0.2),
            A.MotionBlur(p=0.5, blur_limit=25)
        ], p=0.7),
        A.OneOf([
            A.RandomShadow(p=0.5, num_shadows_lower=5, num_shadows_upper=10),
            A.RandomBrightnessContrast(p=0.4, brightness_limit=0.5)
        ], p=0.2),

    ]
    use_train_aug = True
    n_epochs = 45
    early_stopping_epochs = 5
    BATCH_SIZE = 1
    update_every_n_batches = 64
    duplicate_dataset_int_times = 0
    loader_num_workers = 4

    encoder_weights = 'imagenet'
    encoder_name = 'efficientnet-b1'

    model = smp.Unet(encoder_name,
                     activation='logsoftmax',
                     classes=2,
                     encoder_weights=encoder_weights,
                     # decoder_attention_type=None,
                     ).cuda()

    criterion_name = 'SoftDice'
    LR = 3e-3
    criterion = make_criterion(criterion_name=criterion_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=3e-6)
    scheduler_milestones_list = [10]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones_list, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.3,
        div_factor=25,
        max_lr=3,
        epochs=n_epochs,
        steps_per_epoch=64
    )


    inference_model_path = os.path.join(r'train_history\19-09-2022_20-15-06',
                                        'model_checkpoint.pth')

    training_checkpoint_path = os.path.join('train_history/19-09-2022_15-16-43',
                                                 'model_checkpoint.pth')

    #######################################################################
    run_config = {
        'start_time': start_time,
        'BATCH_SIZE': BATCH_SIZE,
        'update_every_n_batches': update_every_n_batches,
        'n_epochs': n_epochs,
        'early_stopping_epochs': early_stopping_epochs,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        #
        'train_files_folder': train_files_folder,
        'original_size': (1232, 1624),
        'patches_size': (224, 224),
        'patches_size_submit': (1344, 1792),
        'non_patches_size': (960, 960),
        'non_patches_size_submit': (1248, 1632),
        #
        'FLAG_PATCHES': False,
        'fix_polygons': True,
        'test_size': test_size,
        'train_aug_list': train_aug_list,
        'use_train_aug': use_train_aug,
        'duplicate_dataset_int_times': duplicate_dataset_int_times,
        'loader_num_workers': loader_num_workers,
        # 'training_checkpoint_path': training_checkpoint_path,
        # 'inference_model_path': inference_model_path,
        'model': model,
    }
    train_folder = f'train_history/{start_time}'
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    with open(os.path.join(train_folder, 'config.txt'), 'w') as f:
        print(run_config,
              file=f)

    main(**run_config)

