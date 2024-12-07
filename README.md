
# Enova Robotic Chat System

This repository contains a test folder for the **Enova Robotic Chat System** with the **Mistral 7B-instruct** model integrated. Additionally, a **Retrieval-Augmented Generation (RAG)** system has been added on top to enhance its capabilities.

## Project Setup

### Python Version
It is recommended to use **Python 3.12.4** for this project to ensure compatibility with all dependencies.

### Creating a Virtual Environment

#### For Windows Users:
To create a virtual environment on Windows, use the following command:

```
C:\Python312\python.exe -m venv chat
```

#### For macOS/Linux Users:
To create a virtual environment on macOS or Linux, use the following command:

```
python3.12.4 -m venv chat
```

This will create a `chat` folder containing the virtual environment.

### Activating the Virtual Environment

After creating the virtual environment, you need to activate it:

#### For Windows:
```
chat\Scripts\activate
```

#### For macOS/Linux:
```
source chat/bin/activate
```

### Installing Required Libraries

Once the virtual environment is activated, install all the necessary libraries by running:

```
pip install -r requirements.txt
```

This will install all the dependencies listed in the `requirements.txt` file.

NB: for torch GPU support make sure you download the right packages version. Check `https://pytorch.org/get-started/locally/` for pytorch and for the CUDA driver you simply follow the steps in `https://developer.nvidia.com/cuda-downloads`. BE CAREFUL WITH CUDA VERSION AND PYTORCH SINCE THEY NEED TO MATCH VERSIONS OTHERWISE CUDA WON'T WORK.

## Project Structure

- **test/**: This folder contains test scripts for the Enova robotic chat system.
- **requirements.txt**: A file listing all the Python libraries required for this project.

## Usage

After setting up the environment and installing dependencies, you can start working with the **Mistral 7B-instruct** model and the integrated RAG system. The model used here is quantized using `BitsandBytes` library to run on low Vram GPU. For more details on this library refer to this link `https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?bnb=8-bit#4-bit-qlora-algorithm`.

# NB

For the models used in this repo, please make sure you download them from huggingface using your own API key before running this repo, run `huggingface-cli login`:
- LLM: `mistralai/Mistral-7B-Instruct-v0.3`
- RAG: `sentence-transformers/all-MiniLM-L6-v2`
- Re-Ranking system (optional): `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Speech-to-Text: `openai/whisper-base`
- Text-to-Speech: `microsoft/speecht5_tts`

# Table of compute using different GPUs
| GPU      | VRAM | TERAFLOPS | Time to run in minuts   |
|-----------|-----|-----|-----------|
| RTX 3050 Mobile|4 GB| 	4.329 | 5 |
| Tesla T4       |16 GB| 	8.141  | 2 |
| A10 G   |12 GB|	31.52 | TBD |

---
Make sure to change the paths for each model in the code as needed.

For the Re-Ranking system it's just used for a better and more accurate retrival. If you are using a larger LLM models you can ignore adding the re-ranking system.

---
For any issues or feature requests, please refer to the project's GitHub Issues page.
