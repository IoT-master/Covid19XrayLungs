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


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = (image - self.mean)/self.std
        return {'image': image, 'label': label}

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

    def __init__(self, dataframe, root_dir, transform):
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

class XRayModel(LightningModule, out1):
    def __init__(self):
        super(XRayModel, self).__init__()
        self.conv1 = nn.Conv2d(1, ou1, 21, 1)
        self.norm1 = nn.BatchNorm2d(out1)
        self.conv2 = nn.Conv2d(out1, 45, 21, 1)  
        self.norm2 = nn.BatchNorm2d(45)
        self.conv3 = nn.Conv2d(45, 30, 6, 1)        
        self.norm3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 25, 6, 1)
        self.norm4 = nn.BatchNorm2d(25)
        self.fc1 = nn.Linear(5**2*25, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [b, 1, 200, 200] ==> [b, 100, 180, 180]
        x = self.norm1(x)
        x = F.max_pool2d(x, 2, 2) # [b, 100, 180, 180] ==> [b, 100, 90, 90]
        x = F.relu(self.conv2(x)) # [b, 100, 90, 90] ==> [b, 75, 70, 70]    
        x = self.norm2(x)    
        x = F.max_pool2d(x, 2, 2) # [b, 75, 70, 70] ==> [b, 75, 35, 35]
        x = F.relu(self.conv3(x)) # [b, 75, 35, 35] ==> [b, 50, 30, 30]
        x = self.norm3(x)
        x = F.max_pool2d(x, 2, 2) # [b, 50, 30, 30] ==> [b, 50, 15, 15]
        x = F.relu(self.conv4(x)) # [b, 50, 15, 15] ==> [b, 25, 10, 10]
        x = self.norm4(x)
        x = F.max_pool2d(x, 2, 2) # [b, 25, 10, 10] ==> [b, 25, 5, 5]
        x = x.view(-1, 5**2*25)
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # # There's no activation at the final layer because of the criterion of CEL
        # return x
        return torch.log_softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        preds = self(images)
        
        # loss = F.cross_entropy(preds, labels)
        loss = F.nll_loss(preds, labels)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)
        # return optim.SGD(self.parameters(), lr=0.01)

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
                ToTensor(),
                Normalize(129.7539, 64.6764)
        ]))
        batch_loader_params = {
            "batch_size": 75 if platform.system() == 'Windows' else 5,
            "shuffle": True,
            "num_workers": 4
        }
        dataloader = DataLoader(my_train_dataset, **batch_loader_params)
        return dataloader

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        preds = self(images)
        return {'val_loss': F.nll_loss(preds, labels)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        import platform
        my_path = "../Datasets/Lungs_Dataset/Xray" if platform.system() == 'Windows' else "datasets/data/images"
        test_df = pd.read_csv('test_df2.csv')

        my_test_dataset = CovidLungsDataset(test_df, my_path, transform=tf.Compose([
                RescaleImage(200),
                ToTensor(),
                Normalize(129.7539, 64.6764)
        ]))
        batch_loader_params = {
            "batch_size": 75 if platform.system() == 'Windows' else 5,
            "shuffle": False,
            "num_workers": 4
        }
        dataloader = DataLoader(my_test_dataset, **batch_loader_params)

        return dataloader

    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        preds = self(images)
        return {'val_loss': F.nll_loss(preds, labels)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):
        import platform
        my_path = "../Datasets/Lungs_Dataset/Xray" if platform.system() == 'Windows' else "datasets/data/images"
        test_df = pd.read_csv('test_df2.csv')

        my_test_dataset = CovidLungsDataset(test_df, my_path, transform=tf.Compose([
                RescaleImage(200),
                ToTensor(),
                Normalize(129.7539, 64.6764)
        ]))
        batch_loader_params = {
            "batch_size": 75 if platform.system() == 'Windows' else 5,
            "shuffle": False,
            "num_workers": 4
        }
        dataloader = DataLoader(my_test_dataset, **batch_loader_params)

        return dataloader
        
if __name__ == "__main__":
    # model = XRayModel()
    model = XRayModel.load_from_checkpoint(checkpoint_path="lightning_logs/version_3/checkpoints/epoch=5.ckpt")
    # trainer = Trainer()
    trainer = Trainer(gpus=1)
    trainer.fit(model)

    # model = XRayModel.load_from_checkpoint(checkpoint_path="lightning_logs/version_3/checkpoints/epoch=5.ckpt")
    # # trainer = Trainer()
    # trainer = Trainer(gpus=1)
    # trainer.test(model=model)

    # import platform
    # my_path = "../Datasets/Lungs_Dataset/Xray" if platform.system() == 'Windows' else "datasets/data/images"
    # test_df = pd.read_csv('test_df2.csv')

    # my_test_set = CovidLungsDataset(test_df, my_path, transform=tf.Compose([
    #         RescaleImage(200),
    #         ToTensor(),
    #         Normalize(129.7539, 64.6764)
    # ]))

    # accuracy = 0
    # for index, sample in enumerate(my_test_set):
    #     sample = my_test_set.__getitem__(index)
    #     image = sample['image']
    #     label = sample['label']
    #     sample_input = image.unsqueeze(0)
    #     # print(sample_input.shape)
    #     pred = model(sample_input)
    #     # print(pred.shape)
    #     # print(torch.argmax(pred).detach(),label)
    #     print(accuracy, index)
    #     if torch.argmax(pred).detach() == label:
    #         accuracy += 1
    # print(accuracy/len(my_test_set))