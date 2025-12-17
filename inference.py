import torch
import torch.nn as nn
from torchvision import models,datasets,transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import eval
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    report = classification_report(all_labels, all_preds, zero_division=0)
    return report

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)


model.fc = nn.Sequential(
    nn.Linear(512, 256), 
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,2)
) 

# model.fc = nn.Sequential(
#     nn.Linear(2048, 512), 
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(512,2)
# ) 
        
    
model = model.to(device)
model.load_state_dict(torch.load('best_update.pth'))
preprocess = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
dataset_test = datasets.ImageFolder('val',transform = preprocess)
loader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

print(evaluate(model,loader_test,device))