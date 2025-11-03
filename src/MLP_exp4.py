import torch 
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as F # <-- Importante

import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os # Necesario

# --- 1. Definición del Modelo MLP (La que tú especificaste) ---
class MLP(nn.Module):
    """
    MLP Experimento de Invarianza (Baseline): 3 capas ocultas
    Arquitectura: 3072 -> 512 -> 256 -> 128 -> 10
    """
    def __init__(self, input_features=3072, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256), # <-- Tu arquitectura
            nn.ReLU(),
            nn.Linear(256, 128),  # <-- Tu arquitectura
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.flatten(x) 
        logits = self.layers(x)
        return logits

# --- 2. Bloque Principal de Ejecución ---
if __name__ == '__main__':

    # --- Parámetros de Entrenamiento ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.2
    RANDOM_SEED = 7

    # --- Configuración del Dispositivo ---
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Usando dispositivo: {device}")

    # Fijar semillas para reproducibilidad
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
    elif device == "mps":
        torch.mps.manual_seed(RANDOM_SEED)

    # --- 3. Preparación de Datos (Lógica de Invarianza) ---
    
    # 1. Transformación de ENTRENAMIENTO (CON augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # 2. Transformación de TEST/VALIDACIÓN (NORMAL)
    transform_test_normal = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 3. Transformación de TEST (TRASLADADO)
    SHIFT_PIXELS = 6 # (Usa el mismo shift que en tu script de CNN)
    transform_test_shifted = transforms.Compose([
        transforms.Lambda(lambda img: F.affine(
            img, 
            angle=0, 
            translate=(SHIFT_PIXELS, SHIFT_PIXELS),
            scale=1.0, 
            shear=[0.0, 0.0],
            fill=0
        )),
        transforms.ToTensor(),
    ])

    # --- Creación de Datasets ---
    train_set_with_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_set_no_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test_normal 
    )
    test_dataset_normal = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test_normal
    )
    test_dataset_shifted = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test_shifted
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    labels = train_set_no_aug.targets
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(
        indices, test_size=VAL_SPLIT, stratify=labels, random_state=RANDOM_SEED
    )
    
    train_dataset = Subset(train_set_with_aug, train_indices)
    val_dataset = Subset(train_set_no_aug, val_indices) # Valida en datos normales

    # --- Creación de DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader_normal = DataLoader(
        test_dataset_normal, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader_shifted = DataLoader(
        test_dataset_shifted, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images:  {len(val_dataset)}")
    print(f"  Test images (Normal): {len(test_dataset_normal)}")
    print(f"  Test images (Shifted): {len(test_dataset_shifted)}")

    # --- 4. Model Initialization ---
    model = MLP().to(device) # <-- Tu Modelo MLP
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # --- 5. Training Loop ---
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_time': []
    }
    best_val_loss = float('inf') 
    best_model_path = 'best_mlp_model_exp4.pth' # Nombre único

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train() 
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss = running_train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # --- Validation Phase ---
        model.eval() 
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        
        epoch_duration = time.time() - epoch_start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['epoch_time'].append(epoch_duration)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Time: {epoch_duration:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            selected_epoch = epoch + 1
            
    total_training_time = time.time() - total_start_time
    
    print(f"\nTraining finished in {total_training_time/60:.2f} minutes.")
    print(f"Best model saved in epoch {selected_epoch}")

    # --- 6. Final Evaluation (Lógica de Invarianza) ---
    print(f"\nLoading best model ('{best_model_path}') for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # --- Evaluación 1: Test Set NORMAL (Baseline) ---
    print("Running evaluation on NORMAL Test Set...")
    all_labels_normal = []
    all_predictions_normal = []
    start_inference_time_normal = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader_normal: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions_normal.extend(predicted.cpu().numpy())
            all_labels_normal.extend(labels.cpu().numpy())
    total_inference_time_normal = time.time() - start_inference_time_normal

    test_accuracy_normal = 100 * (np.array(all_predictions_normal) == np.array(all_labels_normal)).sum() / len(all_labels_normal)
    report_normal_dict = classification_report(all_labels_normal, all_predictions_normal, target_names=classes, output_dict=True)
    report_normal_str = classification_report(all_labels_normal, all_predictions_normal, target_names=classes)
    
    # --- Evaluación 2: Test Set TRASLADADO (Experimento) ---
    print("\nRunning evaluation on SHIFTED Test Set...")
    all_labels_shifted = []
    all_predictions_shifted = []
    start_inference_time_shifted = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader_shifted: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions_shifted.extend(predicted.cpu().numpy())
            all_labels_shifted.extend(labels.cpu().numpy())
    total_inference_time_shifted = time.time() - start_inference_time_shifted

    test_accuracy_shifted = 100 * (np.array(all_predictions_shifted) == np.array(all_labels_shifted)).sum() / len(all_labels_shifted)
    report_shifted_dict = classification_report(all_labels_shifted, all_predictions_shifted, target_names=classes, output_dict=True)
    report_shifted_str = classification_report(all_labels_shifted, all_predictions_shifted, target_names=classes)

    # --- Mostrar Resultados Comparativos ---
    print("\n" + "="*40)
    print(f"TRANSLATION INVARIANCE EXPERIMENT (MLP Baseline)")
    print("="*40)
    print(f"Accuracy on NORMAL Test Set (Baseline): {test_accuracy_normal:.2f} %")
    print(f"Accuracy on SHIFTED Test Set:         {test_accuracy_shifted:.2f} %")
    print("\nReport (NORMAL):")
    print(report_normal_str)
    print("\nReport (SHIFTED):")
    print(report_shifted_str)
    print("="*40)

    # --- 7. Save Results for Notebook --- 
    print("Saving results for notebook analysis...")

    results_filename = 'results_MLP_exp4.pth' # <-- Archivo de salida

    results_to_save = {
        'experiment_name': 'MLP (Baseline) Invariance', # <-- Nombre
        'model_architecture': 'MLP (512-128-64)',
        'history': history,
        'total_params': total_params,
        'total_training_time_sec': total_training_time,
        'classes': classes,
        
        'all_labels_normal': all_labels_normal,
        'all_predictions_normal': all_predictions_normal,
        'test_accuracy_normal': test_accuracy_normal,
        'classification_report_dict_normal': report_normal_dict,
        'classification_report_str_normal': report_normal_str,
        'total_inference_time_sec_normal': total_inference_time_normal,
        
        'all_labels_shifted': all_labels_shifted,
        'all_predictions_shifted': all_predictions_shifted,
        'test_accuracy_shifted': test_accuracy_shifted,
        'classification_report_dict_shifted': report_shifted_dict, # <-- ¡Este es el que busca tu Celda 9!
        'classification_report_str_shifted': report_shifted_str,
        'total_inference_time_sec_shifted': total_inference_time_shifted
    }

    torch.save(results_to_save, results_filename)
    print(f"Results saved to '{results_filename}'.")
    print("Training script finished.")