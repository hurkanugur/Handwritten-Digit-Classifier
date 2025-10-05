import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config

class MNISTDataset:
    def __init__(self):
        """Initialize MNIST dataset transformations with normalization and augmentation."""

        # Training transform
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),                                              # Random rotation for data augmentation
            transforms.RandomAffine(0, translate=(0.1, 0.1)),                           # Random translations for better generalization
            transforms.RandomHorizontalFlip(),                                          # Random horizontal flip for added robustness
            transforms.ToTensor(),                                                      # Convert image to tensor
            transforms.Normalize(mean=(config.MNIST_MEAN,), std=(config.MNIST_STD,))    # Normalization
        ])
        
        # Test transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),                                                      # Convert image to tensor
            transforms.Normalize(mean=(config.MNIST_MEAN,), std=(config.MNIST_STD,))    # Normalization
        ])
        
        # Inference transform
        self.inference_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),                                # Ensure the image is grayscale
            transforms.Resize((28, 28)),                                                # Resize to MNIST size (28x28)
            transforms.ToTensor(),                                                      # Convert image to tensor
            transforms.Lambda(lambda X: 1 - X),                                         # Invert image colors (black -> white, white -> black)
            transforms.Normalize(mean=(config.MNIST_MEAN,), std=(config.MNIST_STD,))    # Normalization
        ])

    # ----------------- Public Methods -----------------

    def get_flattened_input_size(self, data_loader):
        """Return number of input features per sample after flattening (for MLPs)."""
        sample_X, _ = next(iter(data_loader))
        input_dim = sample_X[0].numel()
        print(f"• Input dimension: {input_dim}")
        return input_dim

    def prepare_data_for_training(self):
        """Prepare train, validation, and test DataLoaders from MNIST dataset."""
        train_dataset, test_dataset = self._download_datasets()
        train_dataset, val_dataset = self._split_dataset(train_dataset)
        train_loader, val_loader, test_loader = self._create_data_loaders(train_dataset, val_dataset, test_dataset)

        return train_loader, val_loader, test_loader

    def prepare_data_for_inference(self, image):
        """Prepare image tensor for inference with color inversion."""
        
        # Apply inference transformations (resize, grayscale, normalize...)
        X = self.inference_transform(image)
        
        # Add batch dimension (from [1, 28, 28] to [1, 1, 28, 28])
        X = X.unsqueeze(0)
        
        return X
    
    # ----------------- Private Methods -----------------

    def _download_datasets(self):
        """Download the MNIST datasets for training and testing."""
        train_dataset = datasets.MNIST(config.DATASET_PATH, train=True, download=True, transform=self.train_transform)
        test_dataset = datasets.MNIST(config.DATASET_PATH, train=False, download=True, transform=self.test_transform)
        return train_dataset, test_dataset

    def _split_dataset(self, train_dataset):
        """Split the training dataset into training and validation datasets."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled → using same data for train/val.")
            return train_dataset, train_dataset

        n_train = int(len(train_dataset) * config.TRAIN_SPLIT_RATIO)
        n_val = len(train_dataset) - n_train

        generator = (
            torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED)
            if config.SPLIT_RANDOMIZATION_SEED is not None else None
        )

        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n_val], 
            generator=generator
        )

        return train_dataset, val_dataset      
    
    def _create_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """Return train, val, test DataLoaders."""
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        return train_loader, val_loader, test_loader