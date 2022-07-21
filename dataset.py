from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2


labels = {
    'bacterial_leaf_blight': 0,
    'bacterial_leaf_streak': 1,
    'bacterial_panicle_blight': 2,
    'blast': 3,
    'brown_spot': 4,
    'dead_heart': 5,
    'downy_mildew': 6,
    'hispa': 7,
    'normal': 8,
    'tungro': 9
}

one_hot_encoding = F.one_hot(torch.arange(0, len(labels.keys())) % len(labels.keys()),
                             num_classes=len(labels.keys()))

class PaddyDoctor(Dataset):
    def __init__(self, root_path, x, y, name, transforms=None):
        self.root_path = root_path
        self.images = x
        self.labels = y
        self.transforms = transforms
        self.name = name
        self.get_distribution()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.labels[idx]
        img_path = f'{self.root_path}/{label}/{img_name}'
        img = cv2.imread(img_path)
        img = img.astype('float32')
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        encoded_label = one_hot_encoding[label]
        encoded_label = encoded_label.type(torch.FloatTensor)
        return img, encoded_label

    def get_distribution(self):
        distribution = {}
        for label in self.labels:
            if label not in distribution:
                distribution[label] = 0
            else:
                distribution[label] += 1
        print(f"Distribution of {self.name} dataset: {distribution}")
        return distribution
