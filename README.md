
# **[Ficbot - AI + Inference](https://ficbotweb.com)**

  

_An AI-powered Anime Character Generator for Aspiring Writers_

  

![Anime Character Generator](https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/gifs/namegen.gif)

  

---

  

## **📌 About the Project**

  

Ficbot is a machine learning–powered system designed to assist writers in creating original characters effortlessly. It leverages deep learning and NLP models to generate character names from images, with planned expansions for bio and image generation.

  

### **Project Structure**

Ficbot is now split into two repositories for better organization:

  

- **[ficbot-backend (this repository)](https://github.com/Pythonimous/ficbot-backend)** – Contains:

  - The **AI inference service** that processes images and generates names.

  - The **ML models** and related dependencies (TensorFlow, etc.).

  - Training scripts, dataset processing, and exploratory notebooks.

- **[ficbot](https://github.com/Pythonimous/ficbot)** – Contains:

  - The **frontend** (user interface).

  - The **API layer** that communicates with the backend inference service.

  - Docker configuration for deploying the combined frontend + API container.

  

---

  

#### **🖥 Technical Stack**

  

- **Machine Learning & Inference:**

  - **TensorFlow:** Powers the AI model used for generating character names from images.

  - **FastAPI:** Also used in the backend inference service for serving predictions.

  

- **Frontend & API:**

  - **FastAPI:** Serves the API endpoints used by the frontend.

  - **Bootstrap:** Provides a responsive and modern UI for the web interface.

  - **HTML5/CSS3 & JavaScript:** Standard technologies for building interactive web applications.

  

-  **Deployment & Infrastructure:**
   -  **Docker + AWS Lightsail:** A reliable and cost-effective VPS solution.

  
  

👉 The project originated from **anime character data** on [MyAnimeList](https://myanimelist.net/) and was later expanded to different writing applications.

  

----

  

## **✨ Features**

  

### ✅ **Currently Available**

- **Image → Name Generator:**

Upload an image and get a character name based on AI analysis.

  

### 🚀 **Planned Enhancements**

-  **Additional Name Generators:** (Based on bios and hybrid inputs)
-  **Bio Generators:** (Generate detailed character backstories)
-  **Image Generators:** (AI-generated character visuals)
-  **Anime Filter:** (Transform images into an anime-style character)
-  **Complete OC Generator:** (Generate Name, Bio, and Image together)

  

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

  

### Training and inference

  

1. Training from scratch

```bash
python src/core/train.py --model MODEL_NAME --data_path DATA_PATH --name_col NAME_COL --bio_col BIO_COL --img_col IMG_COL --img_dir IMG_DIR --checkpoint_dir CHECKPOINT_DIR --batch_size 16 --epochs 1 --maxlen 3 --optimizer adam

```

2. Training from checkpoint

```bash
python src/core/train.py --model MODEL_NAME --checkpoint CHECKPOINT_PATH --maps MAPS_PATH --data_path DATA_PATH --name_col NAME_COL --bio_col BIO_COL --img_col IMG_COL --img_dir IMG_DIR --checkpoint_dir CHECKPOINT_DIR --batch_size 16 --epochs 1 --maxlen 3 --optimizer adam

```

 3. Inference

```bash
python src/core/inference.py --model MODEL_NAME --model_path MODEL_PATH --maps MAPS_PATH --img_path IMG_PATH --min_name_length N_WORDS --diversity 1.2
  ```

## 🛠 Docker Deployment

This repository includes a Dockerfile for containerized deployment.

### 1️⃣ Build the Docker Image

```bash
docker build -t img2name .

```

### 2️⃣ Run the Container

```bash
docker run -p 8080:8080 ficbot

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
coverage report # Current coverage: 74%
coverage html -d coverage_html # interactive html reporting

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
