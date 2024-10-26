import lightning as L
import augmentation_lightning
from config_model import early_stopping_callback, check_point_callback
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config_model import LR

torch.set_float32_matmul_precision('medium')

data_module = augmentation_lightning.LightningQrDBest()

L.seed_everything(50)

class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.randn(1, 1, 64, 64)
        self.conv1 = nn.Conv2d(1, 32, 7)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        x = torch.randn((1, 64, 64))
        self._to_linear = None
        
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 4)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=4)
        self.f1_score = torchmetrics.F1Score('multiclass', num_classes=4)
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x.shape[0] * x.shape[1] * x.shape[2]
            
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), 1, 64, 64)
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
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), 1, 64, 64)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LR)
        
num_recognition = Net()

trainer = L.Trainer(accelerator="gpu", devices=[0], precision=16,
                    callbacks=[early_stopping_callback, check_point_callback],
                    max_epochs=50, min_epochs=1,deterministic=True
                    )

trainer.fit(model = num_recognition, datamodule=data_module)

num_recognition.eval()
num_recognition.to_onnx("model.onnx")