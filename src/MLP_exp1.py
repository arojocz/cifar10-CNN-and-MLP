import torch 
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os # Necessary for os.path.exists

# --- 1. MLP Model Definition ---
class MLP(nn.Module):
    """
    MLP Experiment 1 (Baseline): 3 hidden layers (512, 256, 128)
    """
    def __init__(self, input_features=3072, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.flatten(x) 
        logits = self.layers(x)
        return logits

# --- 2. Main Execution Block ---
if __name__ == '__main__':

    # --- Training Parameters ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.2
    RANDOM_SEED = 7

    # --- Device Configuration ---
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
    elif device == "mps": 
        torch.mps.manual_seed(RANDOM_SEED)

    # --- 3. Data Preparation ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set_with_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_set_no_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test 
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    labels = train_set_no_aug.targets
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(
        indices, test_size=VAL_SPLIT, stratify=labels, random_state=RANDOM_SEED
    )

    train_dataset = Subset(train_set_with_aug, train_indices)
    val_dataset = Subset(train_set_no_aug, val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images:  {len(val_dataset)}")
    print(f"  Test images:          {len(test_dataset)}")

    # --- 4. Model Initialization ---
    model = MLP().to(device)
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
    best_model_path = 'best_mlp_model_exp1.pth' # Unique model name

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    total_start_time = time.time() # Start total timer

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

    # --- TOTAL TIME CALCULATION ---
    total_training_time = time.time() - total_start_time
            
    print(f"\nTraining finished in {total_training_time/60:.2f} minutes.")
    print(f"Best model saved to '{best_model_path}'")

    # --- 6. Final Evaluation on Test Set ---
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_labels = []
    all_predictions = []

    start_inference_time = time.time() # Start test timer
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total_inference_time = time.time() - start_inference_time # End timer

    test_accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).sum() / len(all_labels)
    # Generate report as dictionary and string
    report_dict = classification_report(all_labels, all_predictions, target_names=classes, output_dict=True)
    report_str = classification_report(all_labels, all_predictions, target_names=classes)

    print(f"Final TEST Accuracy: {test_accuracy:.2f} %")
    print(f"Total Test Time (Inference): {total_inference_time:.2f}s")
    print("-" * 30)
    print("Classification Report:")
    print(report_str)
    print("-" * 30)

    # --- 7. Save Results for Notebook ---
    results_filename = 'results_MLP_exp1.pth'
    
    print(f"Saving results to '{results_filename}'...")

    results_to_save = {
        'experiment_name': 'MLP Exp 1 (Baseline)',
        'model_architecture': 'MLP (512-256-128)',
        'history': history,
        'all_labels': all_labels,
        'all_predictions': all_predictions,
        'test_accuracy': test_accuracy,
        'classification_report_dict': report_dict,
        'classification_report_str': report_str,
        'total_params': total_params,
        'total_training_time_sec': total_training_time,
        'total_inference_time_sec': total_inference_time,
        'classes': classes
    }

    torch.save(results_to_save, results_filename)
    print(f"Results saved to '{results_filename}'.")
    print("Training script finished.")