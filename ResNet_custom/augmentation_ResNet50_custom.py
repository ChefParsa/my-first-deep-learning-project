import lightning as L
import config_model
from config_model import target_transforms, QrDBest
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.transforms import Compose

class LightningQrDBest(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        
        self.img_dir = config_model.IMAGE_DIR 
        
        self.train_transforms= Compose(config_model.train_transform_compose)
        
        self.val_transforms = Compose(config_model.test_transform_compose)
        
        self.target_transforms = target_transforms
        
        self.batch_size = config_model.BATCH_SIZE
        
    def setup(self, stage: str):
        if stage == 'fit':
            self.full_dataset = QrDBest(self.img_dir)
            self.train_size = int( 0.9 * len(self.full_dataset))
            self.val_size = len(self.full_dataset) - self.train_size
            self.train_dataset, self.val_dataset = random_split(self.full_dataset, [self.train_size, self.val_size],torch.Generator().manual_seed(50))
            self.train_dataset.dataset.img_transforms = self.train_transforms
            self.train_dataset.dataset.target_transforms = self.target_transforms
            self.val_dataset.dataset.img_transforms = self.val_transforms
            self.val_dataset.dataset.target_transforms = self.target_transforms
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=config_model.NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=config_model.NUM_WORKERS)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=config_model.NUM_WORKERS)