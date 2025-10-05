import torch
from torch import nn
from dataset import MNISTDataset
from model import MNISTClassificationModel
from visualize import LossMonitor
from device_manager import DeviceManager
import torch.nn.functional as F
import config

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor):
    """Train a PyTorch model with optional validation and live loss monitoring."""

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # -------------------------
        # Training Step
        # -------------------------
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -------------------------
        # Validation Step
        # -------------------------        
        should_validate = (
            epoch == 1
            or epoch == config.NUM_EPOCHS
            or epoch % config.VAL_INTERVAL == 0
        )

        val_loss = None
        if should_validate:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += loss_fn(outputs, y_batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")
        else:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.8f}")

        # -------------------------
        # Update Training/Validation Loss Graph
        # -------------------------
        loss_monitor.update(epoch, train_loss, val_loss)

def test_model(model, test_loader, device):
    """Evaluate a trained model on the test dataset."""
    
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        # Iterate through the test data
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            probabilities = F.softmax(outputs, dim=1)
            probability, predicted_class_index = torch.max(probabilities, dim=1)

            # Update counters
            total_samples += y_batch.size(0)
            correct_predictions += (predicted_class_index == y_batch).sum().item()

    # Calculate the accuracy
    accuracy = correct_predictions / total_samples

    # Print the final accuracy
    print(f"Test Accuracy: {accuracy:.4f}")

def main():

    # Select CUDA (GPU) / MPS (Mac) / CPU
    device_manager = DeviceManager()
    device = device_manager.device

    # Load and prepare data
    dataset = MNISTDataset()
    train_loader, val_loader, test_loader = dataset.prepare_data_for_training()

    # Model, optimizer, loss function
    input_dim = dataset.get_flattened_input_size(train_loader)
    model = MNISTClassificationModel(input_dim=input_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    # Loss Monitoring
    loss_monitor = LossMonitor()

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor)

    # Test the model
    test_model(model, test_loader, device)

    # Save the model
    model.save()

    # Keep the final plot displayed
    loss_monitor.close()

    # Release the memory
    device_manager.release_memory()

if __name__ == "__main__":
    main()
