import torch
import torch.nn as nn
from torchvision import models,datasets,transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import fit
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# model = models.ResNet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

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


transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=90, expand=False, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


dataset_train = datasets.ImageFolder('train',transform = transform)
dataset_val = datasets.ImageFolder('test',transform = transform)

labels = np.array(dataset_train.targets)
class_counts = np.bincount(labels)
total = sum(class_counts)
class_weights = [total / c for c in class_counts]  
class_weights = torch.FloatTensor(class_weights).to(device)

loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

loss_fn = nn.CrossEntropyLoss(weight= class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,     
    eta_min=1e-5  
)
fit(model,20,loader,loader_val,optimizer,loss_fn,scheduler,device)




