import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix, classification_report

# Configurar logging
test_log_file = "test.log"
logging.basicConfig(filename=test_log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Definir transformaciones para el dataset de prueba
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar dataset de prueba
test_dataset = datasets.ImageFolder(root='dataset/test', transform=test_transform)
workers = os.cpu_count()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=workers)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = models.resnet50()
num_classes = len(test_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_resnet50_transfer_learning.pth'))
model = model.to(device)
model.eval()

# Definir función de pérdida
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Evaluar el modelo
logging.info("Evaluando el modelo en el conjunto de prueba...")
test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calcular métricas
test_accuracy = 100. * test_correct / test_total
logging.info(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Generar y guardar matriz de confusión
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
logging.info("Matriz de confusión guardada como confusion_matrix.png")

# Generar reporte de clasificación
class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
logging.info("Classification Report:\n" + class_report)
print("Classification Report:\n" + class_report)

# Función para extraer activaciones
def get_activations(model, image):
    activation = {}
    def hook_fn(module, input, output):
        activation['features'] = output.detach()
    
    hook = model.layer4.register_forward_hook(hook_fn)
    _ = model(image)
    hook.remove()
    return activation['features']

# Visualizar imágenes y mapas de activación
samples, _ = next(iter(test_loader))
samples = samples[:5].to(device)
activations = get_activations(model, samples)

fig, axes = plt.subplots(5, 2, figsize=(10, 20))
for i in range(5):
    img = samples[i].cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    axes[i, 0].imshow(img)
    axes[i, 0].axis('off')
    axes[i, 0].set_title("Original Image")
    
    activation_map = activations[i].mean(dim=0).cpu().numpy()
    axes[i, 1].imshow(activation_map, cmap='viridis')
    axes[i, 1].axis('off')
    axes[i, 1].set_title("Activation Map")

plt.tight_layout()
plt.show()
logging.info("Visualización de activaciones completada.")
