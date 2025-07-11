# 🦙 Llama-2_from_sratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

> **Full implementation of LLaMA 2 from scratch, featuring Rotary Positional Embedding, RMS Normalization, Multi-Query Attention, KV Cache, Grouped Query Attention (GQA), SwiGLU Activation, and more!**

---

## ✨ Features
- **Rotary Positional Embedding**
- **RMS Normalization**
- **Multi-Query & Grouped Query Attention (GQA)**
- **KV Cache for Fast Inference**
- **SwiGLU Activation Function**
- **From-scratch implementation for learning and research**

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Llama-2_from_sratch.git
   cd Llama-2_from_sratch
   ```
2. **(Optional) Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

### Download Model Weights (LLaMA 2 7B)
```bash
bash download.sh
```

### Run Inference
```bash
python inference.py --prompt "Your prompt here"
```

### Train or Explore the Model
- See `model.py` for model architecture and training logic.

---

## 🧠 Model Details
- **Architecture:** LLaMA 2 (7B parameters)
- **Key Components:**
  - Rotary Positional Embedding
  - RMS Normalization
  - Multi-Query & Grouped Query Attention
  - KV Cache for efficient inference
  - SwiGLU Activation
- **Directory:** `llama-2-7b/` contains model weights/configs and parameters for LLaMA 2 7B

---

## 📁 Directory Structure
```
Llama-2_from_sratch/
├── download.sh           # Script to download model weights
├── inference.py          # Inference script
├── LICENSE               # License file
├── llama-2-7b/           # Model weights/configs for LLaMA 2 7B
├── model.py              # Model architecture & training
├── README.md             # This file
```

---

## 🤝 Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📜 License
This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgements
- [Meta AI](https://ai.facebook.com/research/publications/llama-open-and-efficient-foundation-language-models/) for LLaMA
- [Hugging Face](https://huggingface.co/) for open-source ML tools
- The open-source ML community
