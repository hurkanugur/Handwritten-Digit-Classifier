import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from src.dataset import MNISTDataset
from src.model import MNISTClassificationModel


class InferencePipeline:
    """
    Handles model loading, image preprocessing, and digit classification.
    """

    # ----------------- Initialization -----------------

    def __init__(
            self, 
            model: MNISTClassificationModel, 
            dataset: MNISTDataset, 
            device: torch.device
        ):

        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

    # ----------------- Public Methods -----------------

    def predict(self, image_pixels):
        """
        Perform digit classification on an uploaded MNIST image.
        Returns formatted prediction text.
        """
        if image_pixels is None:
            return "⚠️ Please upload an image first!"

        # Convert to PIL image
        image = Image.fromarray(image_pixels)

        # Preprocess input for inference
        X = self.dataset.prepare_data_for_inference(image).to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)
            confidence = round(confidence.item() * 100, 2)

        return f"Predicted Digit: {predicted_index.item()} — {confidence}% confidence"

    def create_gradio_app(self) -> gr.Blocks:
        """
        Build and return the Gradio interface for interactive inference.
        """
        with gr.Blocks(theme=gr.themes.Ocean(), title="MNIST Digit Classifier") as app:
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
                    analyze_btn = gr.Button("🔍 Classify Digit", variant="primary")
                    clear_btn = gr.Button("🧹 Clear", variant="secondary")

            # Button actions
            analyze_btn.click(
                fn=self.predict,
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
                💡 **Tip:** For the best results, use **black digits** on a **white background**.  
                📊 Model trained on the MNIST handwritten digits dataset.  

                ---
                👨‍💻 **Developed by [Hürkan Uğur](https://github.com/hurkanugur)**  
                🔗 Source Code: [MNIST-Digit-Classifier](https://github.com/hurkanugur/MNIST-Digit-Classifier)
                """
            )

        return app
