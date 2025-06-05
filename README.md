# 🧼 Object Remover with Inpainting

A simple and effective image object removal tool using deep learning. This project uses a JIT-traced PyTorch model for object inpainting, allowing you to upload an image and mask out objects directly from a Gradio interface.



https://github.com/user-attachments/assets/89eaaa36-3b0f-4eed-92e2-b9b81541e6e2




---

## 🚀 Features

- ✅ Object removal using a pre-trained inpainting model
- 🎨 Interactive Gradio UI for masking and previewing results
- 🔍 Image preprocessing utilities for deep learning compatibility
- ⚡ Fast inference with TorchScript JIT model
- 📦 Minimal setup with only essential dependencies

---

## 📁 Project Structure



```
├── app.py                            # Gradio interface for object removal
├── config/
│   └── config.yaml                  # Config for model paths and device
├── models/
│   └── obj_remover.pt               # Downloaded TorchScript model
├── src/
│   ├── components/
│   │   └── objectRemover.py        # Main object remover class
│   └── utils/
│       ├── __init__.py             # Image utilities: normalize, resize, pad, YAML reader, model downloader etc.
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/HSAkash/Remove-Object.git
cd Remove-Object

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Model Download

The model will automatically be downloaded from the [URL](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) specified in `config/config.yaml`. If needed, you can manually place a TorchScript `.pt` model at:

```
models/obj_remover.pt
```

Model URL used (Big-Lama):

```
https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
```

---

## 🖼️ How to Use

### Run the app:

```bash
python app.py
```

### In the Gradio UI:

1. Upload or drag & drop an image.
2. Use the brush tool to mark the object you want to remove.
3. Click "Submit".
4. View the clean, inpainted result instantly.

---

## 🧠 How It Works

Under the hood:

* The image and mask are preprocessed:

  * Converted to CHW format
  * Normalized to `[0, 1]`
  * Padded to dimensions divisible by 8 (for CNN compatibility)
* Model inference is run using a TorchScript version of [Big-Lama](https://github.com/Sanster/lama-cleaner).
* The result is cropped back to original dimensions and rendered.

---

## 📜 License

This project is under the **MIT License**. Check [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments

* [Big-Lama Model](https://github.com/Sanster/models)
* [Gradio](https://www.gradio.app/) for the UI
* [PyTorch](https://pytorch.org/) for the deep learning framework


