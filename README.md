
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

## Project Structure

- **test/**: This folder contains test scripts for the Enova robotic chat system.
- **requirements.txt**: A file listing all the Python libraries required for this project.

## Usage

After setting up the environment and installing dependencies, you can start working with the **Mistral 7B-instruct** model and the integrated RAG system. The model used here is quantized using `BitsandBytes` library to run on low Vram GPU. For more details on this library refer to this link https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?bnb=8-bit#4-bit-qlora-algorithm

---
For any issues or feature requests, please refer to the project's GitHub Issues page.
