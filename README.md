
# **[Ficbot - AI + Inference](https://ficbotweb.com)**

  

_An AI-powered Anime Character Generator for Aspiring Writers_

  

![Anime Character Generator](https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/gifs/namegen.gif)

  

---

  

## **📌 About the Project**

  

Ficbot is a machine learning–powered system designed to assist writers in creating original characters effortlessly. It leverages deep learning and NLP models to generate character names from images, with planned expansions for bio and image generation.

  

### **Project Structure**

Ficbot is now split into two repositories for better organization:

  

- **[ficbot-backend (this repository)](https://github.com/Pythonimous/ficbot-backend)** – Contains:
  - The **AI inference service** that processes images and generates names / bios.
  - The **ML models** and related dependencies (Torch, Transformers).
  - Training scripts, dataset processing, and exploratory notebooks.

- **[ficbot](https://github.com/Pythonimous/ficbot)** – Contains:
  - The **frontend** (user interface).
  - The **API layer** that communicates with the backend inference service.
  - Docker configuration for deploying the combined frontend + API container.

  

---

  

## **🖥 Technical Stack**

- **Machine Learning & Inference:**
  - **PyTorch:** Powers the AI model used for generating character names from images.
  - **Transformers:** Provides state-of-the-art NLP models for bio generation.
  - **Llama.cpp:** Enables efficient CPU-based inference for quantized language models.
  - **FastAPI:** Also used in the backend inference service for serving predictions.

- **Frontend & API:**
  - **FastAPI:** Serves the API endpoints used by the frontend.
  - **Bootstrap:** Provides a responsive and modern UI for the web interface.
  - **HTML5/CSS3 & JavaScript:** Standard technologies for building interactive web applications.

- **Deployment & Infrastructure:**
  - **Docker + AWS Lightsail:** A reliable and cost-effective VPS solution.
---

## 📊 Dataset & Exploratory Notebook  

Ficbot's AI models were trained using a **public dataset** of anime characters, which I compiled and explored in depth.  

🔹 **Dataset on Kaggle:** [MyAnimeList Character Dataset](https://www.kaggle.com/datasets/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [View on Kaggle](https://www.kaggle.com/code/ophelion/myanimelist-dataset-exploratory-notebook)  

This dataset includes **over 106,000 characters**, with names, bios, and images, making it a valuable resource for training NLP models.  

---  

## **✨ Features**

### ✅ **Currently Available**

- **🔹 Image → Name Generator**  
  Upload an image, and the AI will generate a character name based on visual analysis.  
  ✨ *Powered by MobileNetV3 + Bidirectional LSTM.*

- **🔹 Bio Generator**  
  Provide a name, and the AI will generate a detailed character bio.  
  ✨ *Built on DistilGPT2 with sliding window chunking (50–200 tokens per bio), and optimized for CPU inference via LlamaCPP.*

---

### 🚀 **Planned Enhancements**

- **🔹 Advanced Name Generators** – Generate names based on bios and hybrid inputs.  
- **🔹 AI-Powered Image Generation** – Create AI-generated character visuals.  
- **🔹 Anime Filter** – Transform images into an anime-style character.  
- **🔹 Complete OC Generator** – Generate a full original character with Name, Bio, and Image.


---

## **🛠 Installation**

### **1. Create and Activate a Virtual Environment**

**Windows (without WSL)**: [Guide](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html)  
**Linux / Windows (with WSL)**: [Guide](https://www.liquidweb.com/kb/how-to-setup-a-python-virtual-environment-on-windows-10/) 

```bash
python3  -m  venv  venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows 

```

  

### **2. Install Dependencies**

  

```bash
pip install -r requirements.txt

```

  

----------

  

## **🚀 Running the Application**

### Downloading data  

1. Download character links (Selenium)

```bash
python scripts/data/download.py --get_links --link_path LINKS_SAVE_PATH

```

2. Download character data + images (Jikan API)

```bash
python scripts/data/download.py --link_path LINKS_SAVE_PATH --data_path CHARACTERS_SAVE_PATH --img_dir IMAGE_SAVE_DIRECTORY

```

### Training and inference  

1. Training from scratch

```bash
python src/core/train.py --data_path DATA_PATH --img_col IMG_COL --name_col NAME_COL --img_dir IMG_DIR ---batch_size 16 --epochs 1 --maxlen 3 --optimizer adam -checkpoint_dir CHECKPOINT_DIR 

```

2. Training from checkpoint

```bash
python src/core/train.py --model MODEL_NAME --checkpoint CHECKPOINT_PATH --maps MAPS_PATH --data_path DATA_PATH --name_col NAME_COL --bio_col BIO_COL --img_col IMG_COL --img_dir IMG_DIR --checkpoint_dir CHECKPOINT_DIR --batch_size 16 --epochs 1 --maxlen 3 --optimizer adam

```

 3. Inference

```bash
python src/core/inference.py --model MODEL_NAME --model_path MODEL_PATH --img_path IMG_PATH --min_name_length N_WORDS --diversity 1.0
  ```

## 🛠 Docker Deployment

This repository includes a Dockerfile for containerized deployment.

### 1️⃣ Build the Docker Image

```bash
docker build -t img2name .

```

### 2️⃣ Run the Container

```bash
docker run -p 8080:8080 img2name

```  

Once running, you can access ficbot-backend endpoints at server address.
You can test it using the following command:
```bash
curl -X GET "http://localhost:8080/health"

```
You can do inference using the following command:
```bash
curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{"image": "<YOUR_BINARY_IMAGE>", "diversity": 1.2, "min_name_length": 2}'

```

----------

## 💂️ Testing & Development

### Running Unit Tests

```bash
python -m unittest

```

### Checking Test Coverage
```bash
pip install coverage
coverage run -m unittest
coverage report -i

```
## **📌 Contributing**

We welcome contributions!

- Report issues or feature requests via GitHub Issues.
- Fork the repository and submit pull requests for new features or bug fixes.
- Check back for roadmap updates and community discussions.
----------

## **🐝 License**

This project is **open-source** under the BSD-3-Clause license.

----------

## **🔗 Links**
🔹 **Live Demo**: [ficbotweb.com](https://ficbotweb.com)  
🔹 **Ficbot**: [ficbot](https://github.com/Pythonimous/ficbot)  
🔹 **Dataset**: [Kaggle](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [Kaggle](https://www.kaggle.com/code/ophelion/myanimelist-dataset-exploratory-notebook)  
