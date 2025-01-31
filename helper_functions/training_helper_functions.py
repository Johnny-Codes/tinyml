import os
import time
import torch
from helper_functions.general_helper_functions import (
    save_model,
    save_class_labels,
)


def train_one_epoch(model, train_loader, optimizer, device, loss_function):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, device, loss_function):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    return epoch_loss, epoch_acc


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    loss_function,
    training_mode,
):
    val_accuracies = []
    training_metrics = {}

    num_epochs = int(input("Enter the number of epochs: "))
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_stats = {}

        print(f"Epoch {epoch}/{num_epochs - 1}")
        if scheduler is not None:
            learning_rate = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: Learning Rate: {scheduler.get_last_lr()[0]}")
        else:
            learning_rate = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: Learning Rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 10)

        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, loss_function
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            device,
            loss_function,
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        epoch_stats["epoch_duration"] = epoch_duration
        epoch_stats["train_loss"] = train_loss
        epoch_stats["train_acc"] = train_acc.item()
        epoch_stats["val_loss"] = val_loss
        epoch_stats["val_acc"] = val_acc.item()
        epoch_stats["learning_rate"] = learning_rate if learning_rate else None

        if scheduler.__class__.__name__ == "StepLR":
            scheduler.step()
        elif scheduler.__class__.__name__ == "ReduceLROnPlateau":
            scheduler.step(val_loss)

        try:
            if val_acc > max(val_accuracies):
                save_model(
                    model=model,
                    model_name=model_name,
                    epoch=epoch,
                    training_mode=training_mode,
                    val_acc=val_acc,
                )
        except ValueError:
            save_model(
                model=model,
                model_name=model_name,
                epoch=epoch,
                training_mode=training_mode,
                val_acc=val_acc,
            )

        save_class_labels(
            class_labels=model.class_labels,
            path=f"./models/{model_name}/class_labels.json",
        )

        val_accuracies.append(val_acc.item())
        training_metrics[epoch] = epoch_stats

        print(f"Train Loss: {train_loss:.5f} Acc: {train_acc:.5f}")
        print(f"Val Loss: {val_loss:.5f} Acc: {val_acc:.5f}")
        print(f"Epoch Time: {epoch_duration:.2f} seconds")
        print()

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    training_metrics["total_training_time"] = total_training_time
    training_metrics["total_epochs"] = num_epochs

    print(f"Total Training Time: {total_training_time:.2f} seconds")
    return training_metrics
