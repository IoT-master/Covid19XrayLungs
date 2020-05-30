from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms as tf
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.array(image)
        if len(image.shape) > 2:
            image = image[:,:,0]

        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.FloatTensor(image).unsqueeze(0),
#         return {'image': torch.from_numpy(image),                
                'label': torch.from_numpy(np.array(label))}

class RescaleImage(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
            
        The samples coming into this class will have its images reduced assuming
        the input is a h, w, c numpy array
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w))

        return {'image': img, 'label': label}

class CovidLungsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe['Image Index'][idx])
        my_image = Image.open(img_name)        
        if len(my_image.size) > 2:
            assert len(my_image.size) > 2
        row = self.dataframe.iloc[idx]
        label = row['Finding Label']
        sample = {'image': my_image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class XRayModel(LightningModule):
    def __init__(self):
        super(XRayModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5, 1)
        self.conv2 = nn.Conv2d(100, 50, 5, 1)        
        self.conv3 = nn.Conv2d(50, 25, 6, 1)  
        self.fc1 = nn.Linear(25*21*21, 32)
        self.fc2 = nn.Linear(32, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [b, 1, 200, 200] ==> [b, 100, 196, 196]
        x = F.max_pool2d(x, 2, 2) # [b, 100, 196, 196] ==> [b, 100, 98, 98]
        x = F.relu(self.conv2(x)) # [b, 100, 98, 98] ==> [b, 50, 94, 94]
        x = F.max_pool2d(x, 2, 2) # [b, 50, 94, 94] ==> [b, 50, 47, 47]
        x = F.relu(self.conv3(x)) # [b, 50, 47, 47] ==> [b, 25, 42, 42]
        x = F.max_pool2d(x, 2, 2) # [b, 25, 42, 42] ==> [b, 25, 21, 21]
        x = x.view(-1, 25*21*21)  # [b, 25, 42, 42] ==> [b, 25x42x42]
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)
        # # There's no activation at the final layer because of the criterion of CEL
#         return x
        return torch.log_softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        import platform
        my_path = "../Datasets/Lungs_Dataset/Xray" if platform.system() == 'Windows' else "datasets/data/images"
        train_df = pd.read_csv('train_df2.csv')
        filename_list = []
        for root, dirs, files in os.walk(my_path, topdown=True):
            for name in files:
                filename_list.append(name)
        train_df = train_df[train_df['Image Index'].isin(filename_list)]

        my_train_dataset = CovidLungsDataset(train_df, my_path, transform=tf.Compose([
                RescaleImage(200),
                ToTensor()
        ]))
        batch_loader_params = {
            "batch_size": 50,
            "shuffle": True,
            "num_workers": 0 if platform.system() == 'Windows' else 4
        }
        dataloader = DataLoader(my_train_dataset, **batch_loader_params)
        return dataloader
        

model = XRayModel()
trainer = Trainer()
# trainer = Trainer(gpus=1)
trainer.fit(model)

# import platform
# my_path = "../Datasets/Lungs_Dataset/Xray" if platform.system() == 'Windows' else "datasets/data/images"
# train_df = pd.read_csv('train_df2.csv')

# my_train_set = CovidLungsDataset(train_df, my_path, transform=tf.Compose([
#         RescaleImage(200),
#         ToTensor()
# ]))

# sample = my_train_set.__getitem__(0)
# image = sample['image']
# label = sample['label']
# sample_input = image.unsqueeze(0)
# print(sample_input.shape)
# model = XRayModel()
# pred = model(sample_input)
# print(pred.shape)