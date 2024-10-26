import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision.io import read_image


class pre_process_data(Dataset):
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
    

def target_transform(label):
    one_hot = torch.zeros((4))
    one_hot[label - 1] = 1.0
    return one_hot
   
def dataloader_preparation():
    
    dataset_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    #transforms.ToTensor(),
    ])
    
    dataset =pre_process_data(images_dir="QR_d_best",img_transforms=dataset_transform)

    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    
    mean, std = dataset.calculate_mean_std(dataloader)
    
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    
    train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    #transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(15),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Normalize(mean=[mean], std = [std])
    ])

    test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[mean], std=[std])
    ])
    
    train_dataset.dataset.img_transforms = train_transform
    train_dataset.dataset.target_transforms = target_transform
    test_dataset.dataset.img_transforms = test_transform
    test_dataset.dataset.target_transforms = target_transform
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return train_size, test_size, train_loader, test_loader
