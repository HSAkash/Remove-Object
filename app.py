import gradio as gr
import numpy as np
from src.components.objectRemover import ObjectRemover
from src.utils import read_yaml
from pyprojroot import here

# Load configuration
config = read_yaml(here("config/config.yaml").__str__())
# Initialize ObjectRemover with configuration
object_remover = ObjectRemover(**config)


def extract_mask(image_data):
    # Extract background (original image)
    background = image_data["background"]

    # Extract first drawn layer (assume this is the mask layer)
    layer = image_data["layers"][0]  # shape: (H, W, 4) - RGBA

    # # Extract alpha channel as mask
    alpha_channel = layer[:, :, 3]
    mask = (alpha_channel > 0).astype(np.uint8) * 255  # Binary mask
    

    # return background, mask
    rm_img = object_remover.remove_object(background, layer)

    return background, mask, rm_img


with gr.Blocks() as demo:
    with gr.Row():
        im = gr.ImageEditor(
            type="numpy",
            height=1000,
            width=800,
        )

    with gr.Row():
        submit_btn = gr.Button("Submit")

    with gr.Row():
        original_output = gr.Image(label="Original Image")
        mask_output = gr.Image(label="Drawn Mask")
        removed_output = gr.Image(label="Removed Image")

    submit_btn.click(
        extract_mask,
        inputs=im,
        outputs=[original_output, mask_output, removed_output],
    )

if __name__ == "__main__":
    demo.launch()
