import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from tqdm import tqdm
import adamp
from typing import Dict

# Configuración
CONFIG: Dict[str, any] = {
    "seed": 47,
    "dataset_path": "dataset",
    "batch_size_initial": 512,
    "batch_size_finetune": 64,
    "num_workers": os.cpu_count(),
    "num_epochs": 100,
    "finetune_epoch": 4,
    "learning_rate": 3e-3,
    "learning_rate_finetune": 1e-4,
    "weight_decay": 1e-5,
    "t_max": 20,
    "eta_min": 1e-7,
    "label_smoothing": 0.1,
    "patience": 10,
    "min_delta": 0.003,
    "log_dir": "runs/ResNet50-AdamP",
    "model_save_path": "resnet50_transfer_learning.pth",
    "best_model_save_path": "best_resnet50_transfer_learning.pth"
}

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler("training.log"),  # Guardar en archivo
                        logging.StreamHandler()  # Mostrar en consola
                        ])

# Configurar semillas
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

# Transformaciones
def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# Cargar dataset
def get_dataloaders(batch_size: int) -> Dict[str, DataLoader]:
    transforms_dict = get_transforms()
    datasets_dict = {
        "train": datasets.ImageFolder(os.path.join(CONFIG["dataset_path"], "train"), transform=transforms_dict["train"]),
        "val": datasets.ImageFolder(os.path.join(CONFIG["dataset_path"], "val"), transform=transforms_dict["val"])
    }
    return {
        "train": DataLoader(datasets_dict["train"], batch_size=batch_size, shuffle=True, num_workers=CONFIG["num_workers"]),
        "val": DataLoader(datasets_dict["val"], batch_size=batch_size, shuffle=False, num_workers=CONFIG["num_workers"])
    }

def initialize_model(num_classes: int):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparar modelo
dataloaders = get_dataloaders(CONFIG["batch_size_initial"])
num_classes = len(dataloaders["train"].dataset.classes)
model = initialize_model(num_classes)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = adamp.AdamP(model.fc.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["finetune_epoch"], eta_min=CONFIG["learning_rate_finetune"])
scaler = torch.cuda.amp.GradScaler()

# Early Stopping
best_val_loss = float('inf')
epochs_without_improvement = 0

# TensorBoard
writer = SummaryWriter(CONFIG["log_dir"])

for epoch in tqdm(range(CONFIG["num_epochs"]), desc="Training", unit="epoch", colour='yellow'):
    if epoch == CONFIG["finetune_epoch"]:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = adamp.AdamP(model.parameters(), lr=CONFIG["learning_rate_finetune"], weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=CONFIG["eta_min"])
        dataloaders = get_dataloaders(CONFIG["batch_size_finetune"])
        logging.info("Fine-tuning activado: todas las capas desbloqueadas.")
    
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
    
    train_loss = running_loss / len(dataloaders["train"].dataset)
    writer.add_scalar('Training Loss', train_loss, epoch)
    logging.info(f'Epoch [{epoch+1}/{CONFIG["num_epochs"]}] Loss: {train_loss:.4f}')

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
    
    val_loss /= len(dataloaders["val"].dataset)
    val_accuracy = 100. * correct / len(dataloaders["val"].dataset)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
    logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
    
    if best_val_loss - val_loss > CONFIG["min_delta"]:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), CONFIG["best_model_save_path"])
        logging.info(f'Modelo guardado con pérdida: {best_val_loss:.4f}')
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= CONFIG["patience"]:
        logging.info(f'Stop early en epoch {epoch+1}')
        break
    
    scheduler.step()

torch.save(model.state_dict(), CONFIG["model_save_path"])
writer.close()
