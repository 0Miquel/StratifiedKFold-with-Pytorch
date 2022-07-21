from sklearn.model_selection import StratifiedKFold
from dataset import PaddyDoctor
from transforms import train_transforms, val_transforms
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

data = pd.read_csv("../datasets/train.csv")
data = data[['image_id', 'label']].values.tolist()
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

k_folds = 5
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(kfold.split(x, y)):
    print(f'Fold Nº {fold}')
    train_x, train_y = x[train_ids], y[train_ids]
    val_x, val_y = x[val_ids], y[val_ids]

    train_dataset = PaddyDoctor(root_path="../datasets/train_images", x=train_x, y=train_y, name="train", transforms=train_transforms)
    val_dataset = PaddyDoctor(root_path="../datasets/train_images", x=val_x, y=val_y, name="val", transforms=val_transforms)

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=True)
