FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api src/api
COPY src/models/img2name src/models/img2name
COPY src/core/inference.py src/core/inference.py
COPY src/core/utils.py src/core/utils.py

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
