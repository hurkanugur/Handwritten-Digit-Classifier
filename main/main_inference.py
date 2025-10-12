from src.device_manager import DeviceManager
from src.dataset import MNISTDataset
from src.model import MNISTClassificationModel
from src.inference import InferencePipeline


def main():
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    # Load dataset and model
    dataset = MNISTDataset()
    model = MNISTClassificationModel(device=device)
    model.load()

    # Build inference pipeline
    inference = InferencePipeline(model, dataset, device)
    app = inference.create_gradio_app()

    # Launch the app
    app.launch(share=True)

    print("-------------------------------------")
    device_manager.release_memory()
    print("-------------------------------------")


if __name__ == "__main__":
    main()
