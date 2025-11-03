import torch 
torch.set_float32_matmul_precision('high') 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as F

import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os # Necesario

# --- 1. Definición del Modelo CNN ---
class SimpleCNN(nn.Module):
    """
    CNN (Baseline 32, 64, 128) para el experimento de invarianza.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Bloques Convolucionales
        self.conv_layers = nn.Sequential(
            # Bloque 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16

            # Bloque 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
            
            # Bloque 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        )
        
        # Clasificador (Fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# --- 2. Bloque Principal de Ejecución ---
if __name__ == '__main__':

    # --- Parámetros de Entrenamiento ---
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Usando dispositivo: {device}")

    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.2
    RANDOM_SEED = 7

    # Fijar semillas para reproducibilidad
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    if device == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
    elif device == "mps":
        torch.mps.manual_seed(RANDOM_SEED)

    # --- 3. Preparación de Datos ---
    
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
    SHIFT_PIXELS = 5 # Tu valor de 5 píxeles
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

    print(f"  Imágenes de Entrenamiento: {len(train_dataset)}")
    print(f"  Imágenes de Validación:  {len(val_dataset)}")
    print(f"  Imágenes de Test (Normal): {len(test_dataset_normal)}")
    print(f"  Imágenes de Test (Trasladado): {len(test_dataset_shifted)}")

    # --- 4. Inicialización del Modelo, Loss y Optimizador ---
    model = SimpleCNN().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros entrenables: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 5. Bucle de Entrenamiento y Validación ---
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_time': []
    }
    best_val_loss = float('inf') 
    best_model_path = 'best_cnn_model_exp4.pth' 

    print(f"\nIniciando entrenamiento por {NUM_EPOCHS} épocas...")
    total_start_time = time.time() # Inicia contador total

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # ... (Fase de Entrenamiento - sin cambios) ...
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

        # ... (Fase de Validación - sin cambios) ...
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

        # ... (Guardar historial y mostrar - sin cambios) ...
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
              f"Tiempo: {epoch_duration:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            selected_epoch = epoch + 1
            
    # --- CÁLCULO DE TIEMPO TOTAL (CORREGIDO) ---
    total_training_time = time.time() - total_start_time

    print(f"\nEntrenamiento finalizado en {total_training_time/60:.2f} minutos.")
    print(f"Mejor modelo guardado en epoch {selected_epoch}")

    # --- 6. Evaluación Final (CORREGIDO) ---
    print(f"\nCargando el mejor modelo ('{best_model_path}') para la evaluación final...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # --- Evaluación 1: Test Set NORMAL (Baseline) ---
    print("Ejecutando evaluación en Test Set NORMAL...")
    all_labels_normal = []
    all_predictions_normal = []
    
    start_inference_time_normal = time.time() # <-- AÑADIDO
    with torch.no_grad():
        for inputs, labels in test_loader_normal: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions_normal.extend(predicted.cpu().numpy())
            all_labels_normal.extend(labels.cpu().numpy())
    total_inference_time_normal = time.time() - start_inference_time_normal # <-- AÑADIDO

    test_accuracy_normal = 100 * (np.array(all_predictions_normal) == np.array(all_labels_normal)).sum() / len(all_labels_normal)
    report_normal_dict = classification_report(all_labels_normal, all_predictions_normal, target_names=classes, output_dict=True) # <-- AÑADIDO
    report_normal_str = classification_report(all_labels_normal, all_predictions_normal, target_names=classes)
    
    # --- Evaluación 2: Test Set TRASLADADO (Experimento) ---
    print("\nEjecutando evaluación en Test Set TRASLADADO...")
    all_labels_shifted = []
    all_predictions_shifted = []

    start_inference_time_shifted = time.time() # <-- AÑADIDO
    with torch.no_grad():
        for inputs, labels in test_loader_shifted:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions_shifted.extend(predicted.cpu().numpy())
            all_labels_shifted.extend(labels.cpu().numpy())
    total_inference_time_shifted = time.time() - start_inference_time_shifted # <-- AÑADIDO

    test_accuracy_shifted = 100 * (np.array(all_predictions_shifted) == np.array(all_labels_shifted)).sum() / len(all_labels_shifted)
    report_shifted_dict = classification_report(all_labels_shifted, all_predictions_shifted, target_names=classes, output_dict=True) # <-- AÑADIDO
    report_shifted_str = classification_report(all_labels_shifted, all_predictions_shifted, target_names=classes)

    # --- Mostrar Resultados Comparativos ---
    print("\n" + "="*40)
    print(f"RESULTADO DEL EXPERIMENTO DE TRASLACIÓN (CNN)")
    print("="*40)
    print(f"Accuracy en Test NORMAL (Baseline): {test_accuracy_normal:.2f} %")
    print(f"Accuracy en Test TRASLADADO:      {test_accuracy_shifted:.2f} %")
    print("\nReporte (NORMAL):")
    print(report_normal_str)
    print("\nReporte (TRASLADADO):")
    print(report_shifted_str)
    print("="*40)

    # --- 7. Guardar Resultados para el Notebook (CORREGIDO) --- 
    print("Guardando resultados para el análisis en el notebook...")
    
    results_filename = 'results_CNN_exp4.pth' # <-- Nombre de archivo

    results_to_save = {
        'experiment_name': 'CNN (Baseline) Invariance', # <-- Nombre
        'model_architecture': 'CNN (32-64-128)',
        'history': history,
        'total_params': total_params,
        'total_training_time_sec': total_training_time,
        'classes': classes,
        
        # Resultados del test normal (baseline)
        'all_labels_normal': all_labels_normal,
        'all_predictions_normal': all_predictions_normal,
        'test_accuracy_normal': test_accuracy_normal,
        'classification_report_dict_normal': report_normal_dict,
        'classification_report_str_normal': report_normal_str,  
        'total_inference_time_sec_normal': total_inference_time_normal,
        
        # Resultados del test trasladado (experimento)
        'all_labels_shifted': all_labels_shifted,
        'all_predictions_shifted': all_predictions_shifted,
        'test_accuracy_shifted': test_accuracy_shifted,
        'classification_report_dict_shifted': report_shifted_dict, # <-- AÑADIDO
        'classification_report_str_shifted': report_shifted_str,   # <-- Renombrado
        'total_inference_time_sec_shifted': total_inference_time_shifted # <-- AÑADIDO
    }

    torch.save(results_to_save, results_filename)
    print(f"Resultados guardados en '{results_filename}'.")
    print("Script de entrenamiento finalizado.")