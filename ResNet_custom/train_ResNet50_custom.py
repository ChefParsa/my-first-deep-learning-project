
import lightning as L
from torchvision import models
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import augmentation_ResNet50_custom
from config_model import early_stopping_callback, check_point_callback

torch.set_float32_matmul_precision("medium")

class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.randn(1, 1, 64, 64)
        self.model = models.resnet50(weights="IMAGENET1K_V2")
        num_classes = 4
        self.model.conv1 = nn.Conv2d(
        in_channels=1,  # Change input channels to 1 for grayscale images
        out_channels=self.model.conv1.out_channels,
        kernel_size=self.model.conv1.kernel_size,
        stride=self.model.conv1.stride,
        padding=self.model.conv1.padding,
        bias=self.model.conv1.bias
        )
        for parameter in self.model.parameters():
            parameter.requires_grad == False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=4)
        self.f1_score = torchmetrics.F1Score('multiclass', num_classes=4)
        
    def forward(self, x):
        x = self.model(x)
        
        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_function(output, y)
        preds = torch.argmax(output, dim=1)
        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        y = torch.argmax(y, dim=1)
        train_accuracy = self.accuracy(preds, y)
        train_f1_score = self.f1_score(preds, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': train_accuracy, 'train_f1_score': train_f1_score},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        y = torch.argmax(y, dim=1)
        val_accuracy = self.accuracy(preds, y)
        val_f1_score = self.f1_score(preds, y)
        self.log_dict({"val_loss": loss, "val_accuracy": val_accuracy, "val_f1_score": val_f1_score},
                      prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
if __name__ == "__main__":
    dm = augmentation_ResNet50_custom.LightningQrDBest()
    
    model = Net()
    
    L.seed_everything(50)
    
    trainer = L.Trainer(accelerator="gpu", devices=[0], precision=16, callbacks=[early_stopping_callback, check_point_callback],
                        max_epochs=50, min_epochs=1)
    
    trainer.fit(model=model, datamodule=dm)
    
    trainer.test(model=model, dataloaders=dm)
    
