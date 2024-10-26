import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
# learning rate 
LR = 0.001

# batch size for dataloaders
BATCH_SIZE = 32

# mean for transforms.Normalize()
MEAN_FOR_NORMALIZE = 0.61

# std for transforms.Normalize()
STD_FOR_NORMALIZE = 0.14

# num workers for dataloaders
NUM_WORKERS = 4

# path of the image dataset
IMAGE_DIR = "../QR_d_best"

# transforms for training dataset
train_transform_compose = [
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN_FOR_NORMALIZE], std = [STD_FOR_NORMALIZE])  
]

# transforms for validation dataset
test_transform_compose = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN_FOR_NORMALIZE], std=[STD_FOR_NORMALIZE])
]

# transforms for labels
def target_transforms(label, num_classes=4):
    one_hot = torch.zeros((num_classes))
    one_hot[label - 1] = 1.0
    return one_hot

# custom dataset module
class QrDBest(Dataset):
    
    def __init__(self, images_dir, img_transforms=None, target_transforms=None):
        self.images_dir = images_dir
        self.img_transforms = img_transforms
        self.target_transforms = target_transforms
        self.img_labels = os.listdir(self.images_dir)
        
       
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.img_labels[idx])
        image = read_image(img_path)
        image = image.float()
        image = image / 255.0
        label = self.calculate_label(self.img_labels[idx])
        #print(self.img_labels[idx])
       
        if self.img_transforms:
            image = self.img_transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        
        return image, label
        
    def calculate_mean_std(self, dataloader):
        mean = 0.0
        std = 0.0
        total_images_count = 0

        for images, _ in dataloader:
            batch_samples = images.size(0)  # Number of images in the batch
            total_images_count += batch_samples
            
            images = images.view(batch_samples, images.size(1), -1)  # Flatten the image pixels
            mean += images.float().mean(2).sum(0)
            std += images.float().std(2).sum(0)

        mean /= total_images_count
        std /= total_images_count

        return mean.item(), std.item()

    def calculate_label(self, img_basename):
        tmp = img_basename[:len(img_basename)-4].split("_")
        row = tmp[2]
        rod_id = int(tmp[3])
        if row in {"A", "B", "E", "F", "I", "J"}:
            label = 4-(rod_id % 4)
        else:
            label = (rod_id % 4)+1
        return label
    
# early stopping call back
early_stopping_callback = EarlyStopping("val_loss", min_delta=0.008, patience=5, verbose=True, mode="min")

# check point call back
check_point_callback = ModelCheckpoint(
    dirpath="checkpoint/",
    filename="ResNet50 on QR_d_best-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    save_top_k=1
)