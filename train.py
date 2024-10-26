import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import augmentation
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def select_device():
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Running on GPU")
    else:
        device = "cpu"
        print("Running on CPU")
    return device
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 7)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        x = torch.randn((1, 64, 64))
        self._to_linear = None
        
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 4)
        
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
    
class earlyStopping():
    def __init__(self, patience=3, min_delta=0.00):
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_loss = None

    def early_stop(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False

def calculate_accuracy(output, labels):
    predicted_class = torch.argmax(output, dim=1)
    true_class = torch.argmax(labels, dim=1)
    comparision = predicted_class == true_class
    count_true = comparision.sum().item()
    total_train = comparision.numel()
    
    accuracy = round(count_true / total_train, 3)
    
    return accuracy

def train(num_recognition, train_loader, test_loader, writer, device):
    optimizer = optim.Adam(num_recognition.parameters(), lr=0.0001)
    loss_function  = nn.CrossEntropyLoss()
    epochs = 50
    earlystopper = earlyStopping(patience=10, min_delta=0.008)
    best_accuracy = -1
    for epoch in range(epochs):
        num_recognition.train()
        train_running_accuracy = 0
        train_running_loss = 0
        
        for train_images, train_labels in train_loader:
            
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            
            num_recognition.zero_grad()
            #optimizer.zero_grad()
            output = num_recognition(train_images)
            
            loss = loss_function(output, train_labels)
            train_running_loss += loss.item()
            train_running_accuracy += calculate_accuracy(output, train_labels)
            
            loss.backward()
            optimizer.step()
        
        # Average loss and accuracy
        train_loss = train_running_loss / len(train_loader)
        train_accuracy = train_running_accuracy / len(train_loader)
        
        # Log training metrics
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        
        #print(f"epoch: {epoch}, loss: {loss}")
        
        # Evaluate on the test set
        num_recognition.eval()
        test_running_loss = 0.0
        test_running_accuracy = 0.0

        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                outputs = num_recognition(test_images)
                loss = loss_function(outputs, test_labels)
                test_running_loss += loss.item()
                test_running_accuracy += calculate_accuracy(outputs, test_labels)

        # Average loss and accuracy
        test_loss = test_running_loss / len(test_loader)
        test_accuracy = test_running_accuracy / len(test_loader)

        # Log test metrics
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        
        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(num_recognition, "best_model.pth")
            print(f"beast accuracy till now is : {best_accuracy}")
        
        if earlystopper.early_stop(test_loss):
            print(f"early stop at epoch: {epoch + 1}")
            break

def model_performance(num_recognition, test_loader, device):
    metrics = {
    1:{"TP":0, "FP":0, "FN":0}, 
    2:{"TP":0, "FP":0, "FN":0}, 
    3:{"TP":0, "FP":0, "FN":0}, 
    4:{"TP":0, "FP":0, "FN":0}
    }
    
    confusion_matrix = [[0 for i in range(4)] for j in range(4)]

    num_recognition.eval()
    
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            output = num_recognition(test_images)
            predicted = torch.argmax(output, dim=1)
            true_label = torch.argmax(test_labels, dim=1)
            comparision = predicted == true_label
            for i in range(len(comparision)):
                if comparision[i]:
                    metrics[int(predicted[i])+1]["TP"] += 1
                    confusion_matrix[true_label[i]][predicted[i]] += 1
                else:
                    metrics[int(predicted[i])+1]["FP"] += 1
                    metrics[int(true_label[i])+1]["FN"] += 1
                    confusion_matrix[true_label[i]][predicted[i]] += 1
      
    print(metrics)
    
    return metrics


def main():
    
    set_seed(50)
    
    device = select_device()
    
    train_size, test_size, train_loader, test_loader = augmentation.dataloader_preparation()
    
    num_recognition = Net().to(device)

    writer = SummaryWriter("runs/logs")
    
    train(num_recognition, train_loader, test_loader, writer, device)
    writer.close()
    
    model_metrics = model_performance(num_recognition, test_loader, device)
    
if __name__ == "__main__":
    main()    