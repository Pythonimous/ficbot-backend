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
COPY src/models/img2name src/models/img2name
COPY src/models/name2bio/*.py src/models/name2bio/
COPY src/models/name2bio/files src/models/name2bio/files
COPY src/models/utils.py src/models/utils.py

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
