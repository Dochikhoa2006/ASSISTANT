---
tags:
- ONNX
- ONNX Runtime
- code
- nlp
- phi4
- phi4 mini
license: mit
language:
- multilingual
- ar
- zh
- cs
- da
- nl
- en
- fi
- fr
- de
- he
- hu
- it
- ja
- ko
- no
- pl
- pt
- ru
- es
- sv
- th
- tr
- uk
---
# Phi-4-Mini-Instruct ONNX models

## Introduction
This repository hosts the optimized versions of the Phi-4 mini models to accelerate inference with ONNX Runtime.

Optimized models are published here in ONNX format to run with ONNX Runtime on CPU and GPU across devices, including server platforms, Windows, Linux and Mac desktops, and mobile CPUs, with the precision best suited to each of these targets.

Here are some of the optimized configurations we have added:  

1. ONNX model for int4 CPU: ONNX model for CPU and mobile using int4 quantization via RTN.
2. ONNX model for int4 GPU: ONNX model for GPU using int4 quantization via RTN.

## Model Run
You can see how to run examples with ORT GenAI [here](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-tutorial.md)

For CPU:

```bash
# Download the model directly using the Hugging Face CLI
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .

# Install the CPU package of ONNX Runtime GenAI
pip install --pre onnxruntime-genai

# Please adjust the model directory (-m) accordingly
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/common.py -o common.py
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-qa.py -o model-qa.py
python model-qa.py -m cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -e cpu
```

For CUDA:

```bash
# Download the model directly using the Hugging Face CLI
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx --include gpu/* --local-dir .

# Install the CUDA package of ONNX Runtime GenAI
pip install --pre onnxruntime-genai-cuda

# Please adjust the model directory (-m) accordingly
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/common.py -o common.py
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-qa.py -o model-qa.py
python model-qa.py -m gpu/gpu-int4-rtn-block-32 -e cuda
```

For DirectML:

```bash
# Download the model directly using the Hugging Face CLI
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx --include gpu/* --local-dir .

# Install the DML package of ONNX Runtime GenAI
pip install --pre onnxruntime-genai-directml

# Please adjust the model directory (-m) accordingly
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/common.py -o common.py
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-qa.py -o model-qa.py
python model-qa.py -m gpu/gpu-int4-rtn-block-32 -e dml
```


## Model Description
- Developed by: Microsoft
- Model type: ONNX
- License: MIT
- Model Description: This is a conversion of the Phi-4 mini model for ONNX Runtime inference.

**Disclaimer:** Model is only an optimization of the base model, any risk associated with the model is the responsibility of the user of the model. Please verify and test for your scenarios. There may be a slight difference in output from the base model with the optimizations applied.

## Base Model
Phi-4-Mini is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data. The model belongs to the Phi-4 model family and supports 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning and direct preference optimization to support precise instruction adherence and robust safety measures.
See details at [https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/README.md](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/README.md)

## Performance Comparison
| Hardware | ONNX | PyTorch | speedup |
| -------|----------|------|---------|
| RTX 4090 GPU | int4: 260.045 tokens/sec fp16: 97.463 tokens/se fp32: 19.320 tokens/sec | fp16: 43.957 tokens/sec | 5x(fp16) |
| Intel Xeon Platinum 8272CL CPU | int4: 16.89 tokens/sec | fp32: 1.636 tokens/sec | 10x |
| Intel Xeon Platinum 8573B CPU | int4: 23.978 tokens/sec | fp32: 4.479 tokens/sec | 5.35X |
| AMD EPYC 7763v CPU | int4: 19.884 tokens/sec | fp32: 1.599 tokens/sec | 12.4x |
| Intel Core Ultra 7 165H Laptop CPU | int4: 4.863 tokens/sec | fp32: 1.699 tokens/sec | 2.8x |
| Intel i7 processor | int4: 3.474 tokens/sec fp32: 1.800 tokens/sec | fp32: 0.702 tokens/sec | 4.85x|