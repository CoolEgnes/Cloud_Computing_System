import azure.functions as func
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import time
import json
import sys
from datetime import datetime

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="resnet")
def resnet(req: func.HttpRequest) -> func.HttpResponse:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # def train_model():
    # Load the full CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Create a small subset (e.g., 1,000 images)
    subset_size = 1000
    indices = torch.randperm(len(full_train_dataset))[:subset_size]  # Randomly select 1,000 images
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load a pre-trained ResNet-50 model
    model = resnet50(pretrained=True)

    # Modify the final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 5  # Fewer epochs for a small dataset
    start_time = time.time()  # Record the start time
    formatted_start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    total_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                total_loss += running_loss
                running_loss = 0.0
    # Calculate total training time
    end_time = time.time()
    training_time = end_time - start_time

    # Save the model
    model_path = 'resnet50_cifar10_subset.pth'
    torch.save(model.state_dict(), model_path)

    print(f"start_time: {formatted_start_time}, total_loss: {total_loss}, training_time: {training_time}, model_path: {model_path}")
    return func.HttpResponse(
        f"start_time: {formatted_start_time}, total_loss: {total_loss}, training_time: {training_time}, model_path: {model_path}",
        status_code=200
    )

    # Return results as a dictionary

    # logging.info('Python HTTP trigger function processed a request.')

    # name = req.params.get('name')
    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    # if name:
    #     return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    # else:
    #     return func.HttpResponse(
    #          "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
    #          status_code=200
    #     )