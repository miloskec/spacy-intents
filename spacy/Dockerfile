FROM huggingface/transformers-pytorch-gpu:latest

# Install dependencies
WORKDIR /workspace

# Install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install python-dotenv langchain accelerate && \
    python3.10 -m pip install flask langchain-community langchain-huggingface && \
    python3.10 -m pip install llama-index llama-index-embeddings-huggingface && \
    python3.10 -m pip install peft auto-gptq optimum bitsandbytes sentence-transformers && \
    python3.10 -m pip install numpy fastapi dataset cupy-cuda12x spacy[cuda12x] uvicorn thinc && \
    python3.10 -m spacy download en_core_web_trf && \
    python3.10 -m spacy download en_core_web_sm && \
    python3.10 -m spacy download en_core_web_md
