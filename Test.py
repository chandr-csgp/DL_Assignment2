import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, alexnet, mobilenet_v2
from torchvision.models import ResNet18_Weights, AlexNet_Weights, MobileNet_V2_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and transformation with different augmentation options
def load_data(batch_size=128, data_augmentation=False, model_name="resnet18"):
    transform_train = [
        transforms.Resize(224) if model_name == "alexnet" else transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4911, 0.4722, 0.4565), (0.2023, 0.1994, 0.2010))
    ]
    if data_augmentation:
        transform_train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + transform_train
    transform_train = transforms.Compose(transform_train)
    
    transform_test = transforms.Compose([
        transforms.Resize(224) if model_name == "alexnet" else transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4911, 0.4722, 0.4565), (0.2023, 0.1994, 0.2010))
    ])

    # Download and load CIFAR-10 training and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders with the specified batch size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

# Model Selection: Choose ResNet-18, AlexNet, or MobileNet
def get_model(model_name="resnet18"):
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 (10 classes)
    elif model_name == "alexnet":
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, 10)
    elif model_name == "mobilenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    else:
        raise ValueError("Invalid model name. Choose 'resnet18', 'alexnet', or 'mobilenet'.")
    return model.to(device)

# Training and validation function with metrics tracking and confusion matrix
def train_model(model, trainloader, valloader, optimizer, criterion, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_f1_scores, val_precisions, val_recalls = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(100 * correct / total)

        # Validation phase with additional metrics (F1-score, precision, recall)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_losses.append(val_loss / len(valloader))
        val_accuracies.append(100 * correct / total)
        val_f1_scores.append(f1_score(y_true, y_pred, average="weighted"))
        val_precisions.append(precision_score(y_true, y_pred, average="weighted"))
        val_recalls.append(recall_score(y_true, y_pred, average="weighted"))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, '
              f'Val Acc: {val_accuracies[-1]:.2f}%, Val F1: {val_f1_scores[-1]:.4f}, '
              f'Val Precision: {val_precisions[-1]:.4f}, Val Recall: {val_recalls[-1]:.4f}')

    # Plotting training and validation metrics
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_f1_scores, label='Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1-Score')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Precision and Recall')

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_accuracies, val_accuracies, val_f1_scores, val_precisions, val_recalls

# Final test confusion matrix
def test_confusion_matrix(model, testloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for Test Set")
    plt.show()

# Main function to run the model training with hyperparameter tuning
def main(model_name="resnet18", batch_size=128, learning_rate=0.001, optimizer_type="adam", num_epochs=10, data_augmentation=True):
    trainloader, valloader, testloader = load_data(batch_size=batch_size, data_augmentation=data_augmentation, model_name=model_name)
    model = get_model(model_name=model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == "adam" else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    train_model(model, trainloader, valloader, optimizer, criterion, num_epochs=num_epochs)
    test_confusion_matrix(model, testloader)

if __name__ == "__main__":
    main(model_name="resnet18", batch_size=64, learning_rate=0.001, optimizer_type="adam", num_epochs=10, data_augmentation=True)

