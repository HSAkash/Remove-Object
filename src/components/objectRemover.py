import os
import time
import torch
import numpy as np
import cv2
from src.utils import (
    download_model,
    norm_img,
    resize_max_size,
    pad_img_to_modulo
)

class ObjectRemover:
    """
    A class for removing objects from images using a pre-trained PyTorch model.

    This class loads a JIT-traced PyTorch model for image inpainting (object removal),
    prepares the input image and mask, and performs inference to generate the inpainted result.

    Attributes:
        model_path (str): Path to the model file (.pt).
        model_url (str): Optional URL to download the model if not already available.
        device (str): Device to use for inference ('cpu' or 'cuda').
    """

    def __init__(self, model_path: str, model_url: str = None, device: str = "cpu"):
        """
        Initializes the ObjectRemover and loads the model.

        Args:
            model_path (str): Local path where the model should be saved/loaded.
            model_url (str, optional): URL to download the model from if not found locally.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device = device
        self.model_path = model_path
        self.model_url = model_url
        self.load_model()

    def load_model(self):
        """
        Loads the PyTorch JIT-traced model from disk or downloads it if a URL is provided.

        Raises:
            FileNotFoundError: If the model does not exist and no URL is provided.
        """
        if self.model_url:
            download_model(self.model_url, self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

    def process_image(self, image: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """
        Preprocesses an input image or mask for model inference.

        Steps:
            - For masks, extracts the alpha channel (if applicable).
            - Resizes the image to fit within its max side.
            - Normalizes and pads the image to ensure dimensions are divisible by 8.

        Args:
            image (np.ndarray): Input image or mask.
            is_mask (bool, optional): Flag to indicate if the input is a mask. Defaults to False.

        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W).
        """
        if is_mask:
            image = image[:, :, 3]  # Extract alpha channel from mask

        size_limit = max(image.shape)  # Limit resize to the longest edge
        image = resize_max_size(image, size_limit=size_limit)
        image = norm_img(image)
        image = pad_img_to_modulo(image, mod=8)

        if is_mask:
            image = (image > 0) * 1  # Binary mask

        image = torch.from_numpy(image).unsqueeze(0).to(self.device, dtype=torch.float32)
        return image

    def remove_object(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Performs object removal (inpainting) on the given image using the mask.

        Args:
            image (np.ndarray): Input image (should be RGBA).
            mask (np.ndarray): Corresponding mask (where objects to be removed are marked in alpha).

        Returns:
            np.ndarray: Inpainted image with objects removed.

        Notes:
            - Outputs are clipped and converted to uint8 RGB.
            - Prints processing time in milliseconds for performance monitoring.
        """
        start_time = time.time()

        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Remove alpha channel from input image
        origin_height, origin_width = image.shape[:2]

        image_tensor = self.process_image(image)
        mask_tensor = self.process_image(mask, is_mask=True)

        with torch.no_grad():
            output = self.model(image_tensor, mask_tensor)

        output = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        output = output[:origin_height, :origin_width, :]  # Crop to original size
        output = np.clip(output * 255, 0, 255).astype("uint8")

        print(f"Object removal took {(time.time() - start_time) * 1000:.2f} ms")
        return output
