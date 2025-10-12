# -------------------------
# Paths
# -------------------------
MODEL_PATH = "model/mnist_digit_classifier.pth"
DATASET_PATH = "data/"
TEST_DATA_FOLDER_PATH = "data/test_data"

# -------------------------
# MNIST mean and std values
# -------------------------
MNIST_MEAN = 0.5
MNIST_STD = 0.5

# -------------------------
# Training hyperparameters
# -------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 20
VAL_INTERVAL = 5

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True
TRAIN_SPLIT_RATIO = 0.9
VAL_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.0                # MNIST dataset provides a dedicated separate test set, so no need for splitting.
SPLIT_RANDOMIZATION_SEED = None       # Int -> Reproducible splits | None -> Fully random splits
