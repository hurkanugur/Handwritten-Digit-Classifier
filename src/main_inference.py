import torch
import config
import torch.nn.functional as F
from dataset import MNISTDataset
from PIL import Image
from device_manager import DeviceManager
from model import MNISTClassificationModel
import os

def main():
    # -------------------------
    # Select CUDA (GPU) / MPS (Mac) / CPU
    # -------------------------
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    # -------------------------
    # Load dataset normalization params and categorical mappings
    # -------------------------
    dataset = MNISTDataset()

    # -------------------------
    # Load trained model
    # -------------------------
    model = MNISTClassificationModel(input_dim=784, device=device)
    model.load()
    model.eval()

    # -------------------------
    # Perform inference on images from 0.png to 9.png
    # -------------------------
    print("-------------------------------------")
    for i in range(10):
        image_path = f"{config.TEST_DATA_FOLDER_PATH}/{i}.png"

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found!")
            continue
        
        # Prepare the image for inference
        image = Image.open(image_path)
        X = dataset.prepare_data_for_inference(image)

        # -------------------------
        # Model inference
        # -------------------------
        X = X.to(device)
        with torch.no_grad():
            outputs = model(X)
            probabilities = F.softmax(outputs, dim=1)
            probability, predicted_class_index = torch.max(probabilities, dim=1)

        # -------------------------
        # Display predictions
        # -------------------------
        print(f"• Image: {os.path.basename(image_path)}")
        print(f"• Predicted Class: {predicted_class_index.item()} (Probability: {probability.item():.4f})")
        print("-------------------------------------")
        # -------------------------
        # Release the memory
        # -------------------------
        device_manager.release_memory()
    
        print("-------------------------------------")

if __name__ == "__main__":
    main()
