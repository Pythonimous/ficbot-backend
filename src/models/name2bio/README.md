# Converting the Model to GGUF for CPU Inference

## 1. Clone llama.cpp Repository
```sh
git clone https://github.com/ggml-org/llama.cpp.git
```

## 2. Install llama.cpp Requirements
```sh
pip install -r llama.cpp/requirements.txt
```

## 3. Convert the Model
Run the following command selecting the folder where the model files are saved:
```sh
python llama.cpp/convert_hf_to_gguf.py MODEL_FOLDER --outfile OUT_PATH --outtype q8_0
```