import torch
import torch.nn.functional as F
from dataset import MNISTDataset
from PIL import Image
from device_manager import DeviceManager
from model import MNISTClassificationModel
import gradio as gr


# -------------------------
# Classification function
# -------------------------
def classify_image(model, dataset, device, image_pixels):
    if image_pixels is None:
        return "⚠️ Please select an image first!"

    image = Image.fromarray(image_pixels)
    X = dataset.prepare_data_for_inference(image)
    X = X.to(device)

    with torch.no_grad():
        outputs = model(X)
        probabilities = F.softmax(outputs, dim=1)
        probability, predicted_class_index = torch.max(probabilities, dim=1)
        probability = round(probability[0].item() * 100, 2)

    return f"Predicted Digit: {predicted_class_index.item()} — {probability}% confidence"

# -------------------------
# Gradio UI Builder
# -------------------------
def create_gradio_app(model, dataset, device):
    with gr.Blocks(theme=gr.themes.Ocean(), title="MNIST Digit Classifier") as demo:
        gr.Markdown(
            """
            # ✨ MNIST Digit Classifier  
            Upload a handwritten digit image, and the model will predict which digit it represents.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="📸 Upload Handwritten Digit", image_mode="RGB")

            with gr.Column(scale=1):
                gr.Markdown("### 🧾 Prediction Result")
                output_text = gr.Textbox(
                    label="Classification",
                    placeholder="Prediction will appear here...",
                    interactive=False,
                    lines=2,
                    show_copy_button=True,
                )
                submit_btn = gr.Button("🔍 Classify Digit", variant="primary", scale=1)
                clear_btn = gr.Button("🧹 Clear", variant="secondary")

        submit_btn.click(
            fn=lambda img: classify_image(model, dataset, device, img),
            inputs=image_input,
            outputs=output_text,
        )

        clear_btn.click(
            fn=lambda: (None, ""),
            inputs=None,
            outputs=[image_input, output_text],
        )

        gr.Markdown(
            """
            ---
            💡 **Tip:** For the best results, use **black digits** on a **white background**.<br>
            📊 Model is trained on the MNIST handwritten digits dataset.  

            ---
            👨‍💻 **Developed by [Hürkan Uğur](https://github.com/hurkanugur)**  
            🔗 Source Code: [MNIST-Digit-Classifier](https://github.com/hurkanugur/MNIST-Digit-Classifier)
            """
        )

    return demo


# -------------------------
# Main entry point
# -------------------------
def main():
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    dataset = MNISTDataset()

    model = MNISTClassificationModel(input_dim=784, device=device)
    model.load()
    model.eval()

    demo = create_gradio_app(model, dataset, device)
    demo.launch(share=True)

    print("-------------------------------------")
    device_manager.release_memory()
    print("-------------------------------------")


if __name__ == "__main__":
    main()
