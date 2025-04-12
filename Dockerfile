FROM public.ecr.aws/docker/library/python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# Set compiler environment variables
ENV CC=gcc
ENV CXX=g++

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api src/api
COPY src/models/utils.py src/models/

#img2name
COPY src/models/img2name/img2name.py src/models/img2name/
COPY src/models/img2name/inference.py src/models/img2name/
RUN mkdir -p /app/src/models/img2name/files && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Pythonimous/ficbot-img2name', filename='files/maps.pkl', local_dir='/app/src/models/img2name/')" && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Pythonimous/ficbot-img2name', filename='files/params.pkl', local_dir='/app/src/models/img2name/')" && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Pythonimous/ficbot-img2name', filename='files/weights.pt', local_dir='/app/src/models/img2name/')"

# name2bio
COPY src/models/name2bio/*.py src/models/name2bio/
RUN mkdir -p /app/src/models/name2bio/files && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Pythonimous/ficbot-name2bio', filename='name2bio.gguf', local_dir='/app/src/models/name2bio/files/')"
COPY src/models/name2bio/files/name2bio_embeddings src/models/name2bio/files/name2bio_embeddings

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
