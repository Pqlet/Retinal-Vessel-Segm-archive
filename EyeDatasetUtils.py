from torch.utils.data import Dataset
import glob
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import json


class EyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """
    def __init__(self, data_folder: str, transform = None, return_filepath=False, is_test_dataset=False, fix_polygons=False):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.transform = transform

        if is_test_dataset:
            self._image_files = (glob.glob(f"{data_folder}/*.png"))
        else:
            self._image_files = set([path.split('.')[0] for path in (glob.glob(f"{data_folder}/*.png"))])
            geojson_files = set([path.split('.')[0] for path in (glob.glob(f"{data_folder}/*.geojson"))])
            self._image_files = list(map(lambda x: x+'.png',
                                         list(self._image_files.intersection(geojson_files)))
                                     )

        self.return_filename = return_filepath
        self.is_test_dataset = is_test_dataset

        self.fix_polygons = fix_polygons

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    @staticmethod
    def parse_polygon(coordinates: dict, image_size: tuple, fix_polygons: bool) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.float32)
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            # FIX OF FILLED POLYGONS
            n_polygon = 1
            for polygon in coordinates:
                points = np.int32(np.array([polygon]))
                fill_value_polygon = 1 if n_polygon == 1 else 0
                cv2.fillPoly(mask, points, fill_value_polygon)
                if fix_polygons:
                    n_polygon += 1
        return mask

    @staticmethod
    def parse_mask(shape: dict, image_size: tuple, fix_polygons: bool) -> np.ndarray:
        """
        Метод для парсинга фигур из geojson файла
        """
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size, fix_polygons)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size, fix_polygons)

        return mask

    def read_layout(self, path: str, image_size: tuple) -> np.ndarray:
        """
        Метод для чтения geojson разметки и перевода в numpy маску
        """
        with open(path, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
            json_contents = json.load(f)

        num_channels = 1 + max(self.class_ids.values())
        mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.float32)

        if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            channel_id = self.class_ids["vessel"]
            mask = self.parse_mask(shape['geometry'], image_size, fix_polygons=self.fix_polygons)
            # THIS IS THE FIX FOR THE MASKS
            mask = np.clip(mask, 0, 1) 
            mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)

        mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)

        return np.stack(mask_channels, axis=-1)

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self._image_files[idx]
        return self.get_item_by_filename(image_path)

    def get_item_by_filename(self, image_path):
        image = self.read_image(image_path)
        mask = None
        sample = {'image': image,
                  'mask': mask}

        if not self.is_test_dataset:
            # Получаем соответствующий файл разметки
            json_path = image_path.replace("png", "geojson")
            sample['mask'] = self.read_layout(json_path, image.shape[:2])
            if self.transform is not None:
                sample = self.transform(**sample)
        else:
            if self.transform is not None:
                sample = self.transform(image=sample['image'])

        if self.return_filename:
            return sample, image_path
        else:
            return sample

    def __len__(self):
        return len(self._image_files)

    # Метод для проверки состояния датасета
    def make_report(self):
      reports = []
      if (not self.data_folder):
        reports.append("Путь к датасету не указан")
      if (len(self._image_files) == 0):
        reports.append("Изображения для распознавания не найдены")
      else:
        reports.append(f"Найдено {len(self._image_files)} изображений")
      cnt_images_without_masks = sum([1 - len(glob.glob(filepath.replace("png", "geojson"))) for filepath in self._image_files])
      if cnt_images_without_masks > 0:
        reports.append(f"Найдено {cnt_images_without_masks} изображений без разметки")
      else:
        reports.append(f"Для всех изображений есть файл разметки")
      return reports

class DatasetPart(Dataset):
    """
    Обертка над классом датасета для его разбиения на части
    """
    def __init__(self, dataset: Dataset,
                 indices: np.ndarray,
                 transform: A.Compose = None,
                 duplicate_dataset_int_times: int = 0,
                 aug: A.Compose = None
                 ):
        self.dataset = dataset
        self.indices = indices
        self._image_files = np.array(self.dataset._image_files)[self.indices]
        self._image_files = np.tile(self._image_files, duplicate_dataset_int_times+1)

        self.transform = transform
        self.aug = aug

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset.get_item_by_filename(self._image_files[idx])

        if self.transform is not None:
            sample = self.transform(**sample)

        if self.aug is not None:
            sample = self.aug(**sample)
            sample = ToTensorV2(transpose_mask=True)(**sample)

        return sample

    def __len__(self) -> int:
        return len(self._image_files)
