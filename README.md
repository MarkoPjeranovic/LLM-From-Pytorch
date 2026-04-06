# What is this?
---
A Pytorch implementation for an LLM.
Planned to be a testing ground for different architectures as well as having basic training and inference capabilities.

# How do I run it?
---
Clone the repo with `git clone https://github.com/MarkoPjeranovic/LLM-From-Pytorch`

This is inteded to be used with a CUDA-capable device (NVIDIA GPU or similar), but can run on the CPU. If running on the CPU, do `pip install -r requirements.txt` or `uv pip install requirements.txt`, inside a virtual environment.

To work with a GPU, you'll need to have CUDA toolkit installed, followed by a matching Pytorch version. For CUDA Toolkit v12.1, the command would be:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## TODO:
Make a huggingface repo with working models
Different attention implementations other than GQA