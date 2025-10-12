import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from src import config
from src.model import MNISTClassificationModel
from src.visualize import LossMonitor

class TrainingPipeline:
    """Handles training, validation, and testing of a model."""

    # ----------------- Initialization -----------------

    def __init__(
        self, 
        model: MNISTClassificationModel, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        loss_monitor: LossMonitor,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.loss_monitor = loss_monitor

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.loss_fn = nn.CrossEntropyLoss()

    # ----------------- Public Methods -----------------

    def train(self):
        """Train the model with optional validation."""
        print("• Training the model:")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            # -------------------------
            # Training Step
            # -------------------------
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch} Training"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            # -------------------------
            # Validation Step
            # -------------------------
            val_loss = None
            if epoch == 1 or epoch == config.NUM_EPOCHS or epoch % config.VAL_INTERVAL == 0:
                val_loss = self._validate(epoch=epoch)
                print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.6f}")

            # -------------------------
            # Update Training Graph
            # -------------------------
            self.loss_monitor.update(epoch, train_loss, val_loss)


    def test(self) -> float:
        """Evaluate a trained model on the test dataset."""
        print("• Testing the model:")
        self.model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.test_loader, desc="Testing"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_class_index = torch.max(probabilities, dim=1)
                total_samples += y_batch.size(0)
                correct_predictions += (predicted_class_index == y_batch).sum().item()

        accuracy = correct_predictions / total_samples
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        return accuracy

    # ----------------- Private Methods -----------------

    def _validate(self, epoch: int) -> float | None:
        """Run model validation."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.val_loader, desc=f"Epoch {epoch} Validating"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                val_loss += self.loss_fn(outputs, y_batch).item()
        return val_loss / len(self.val_loader)
