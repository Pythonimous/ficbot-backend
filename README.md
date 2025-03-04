
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

  

## **🖥 Technical Stack**

  

- **Machine Learning & Inference:**

  - **TensorFlow:** Powers the AI model used for generating character names from images.

  - **FastAPI:** Also used in the backend inference service for serving predictions.

  

- **Frontend & API:**

  - **FastAPI:** Serves the API endpoints used by the frontend.

  - **Bootstrap:** Provides a responsive and modern UI for the web interface.

  - **HTML5/CSS3 & JavaScript:** Standard technologies for building interactive web applications.

  

-  **Deployment & Infrastructure:**
   -  **Docker + AWS Lightsail:** A reliable and cost-effective VPS solution.

---

## 📊 Dataset & Exploratory Notebook  

Ficbot's AI models were trained using a **public dataset** of anime characters, which I compiled and explored in depth.  

🔹 **Dataset on Kaggle:** [MyAnimeList Character Dataset](https://www.kaggle.com/datasets/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [View on Kaggle](https://www.kaggle.com/code/ophelion/myanimelist-dataset-exploratory-notebook)  

This dataset includes **over 106,000 characters**, with names, bios, and images, making it a valuable resource for training NLP models.  

---  

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

### Downloading data  

1. Download character links (Selenium)

```bash
python scripts/data/download.py --get_links --link_path path/to/save/links.txt

```

2. Download character data + images (Jikan API)

```bash
python scripts/data/download.py --link_path path/to/saved/links.txt --data_path path/to/save/data.csv --img_dir path/to/save/images

```

### Training and inference  

#### Img2name model

1. Training from scratch

```bash
python src/models/img2name/train.py --data_path path/to/your/data.csv --name_col NAME_COL --img_col IMG_COL --img_dir path/to/your/images --checkpoint_dir path/to/save/checkpoints --batch_size 16 --epochs 1 --maxlen 3
```
2. Training from checkpoint

```bash
python src/models/img2name/train.py --checkpoint path/to/your/checkpoint.pt --maps path/to/your/maps.pkl --data_path path/to/your/data.csv --name_col NAME_COL --img_col IMG_COL --img_dir path/to/your/images --checkpoint_dir path/to/save/checkpoints --batch_size 16 --epochs 1 --maxlen 3
```

3. Inference

```bash
python src/models/img2name/inference.py --model_path path/to/your/model --img_path path/to/your/image.jpg --min_name_length 2 --diversity 1.2
```

#### Name2bio model

1. Training from scratch

```bash
python src/models/name2bio/train.py --csv_path path/to/your/data.csv --output_dir path/to/save/checkpoints --num_train_epochs 10
```
2. Training from checkpoint

```bash
python src/models/name2bio/train.py --csv_path path/to/your/data.csv --output_dir path/to/save/checkpoints --checkpoint path/to/saved/checkpoint --num_train_epochs 10
```

3. Inference

```bash
python src/models/name2bio/inference.py 'John Doe' --temperature 1.0 --min_length 50 --max_length 200
```

## 🛠 Docker Deployment

This repository includes a Dockerfile for containerized deployment.

### 1️⃣ Build the Docker Image

```bash
docker build -t inference .

```

### 2️⃣ Run the Container

```bash
docker run -p 8080:8080 inference
```

Once running, you can access ficbot-backend endpoints at server address.
You can test it using the following commands:
```bash
curl -X GET "http://localhost:8080/health"

curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{"image": "<YOUR_BINARY_IMAGE>", "diversity": 1.2, "min_name_length": 2, "type": "name"}'

curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{"name": "John Doe", "diversity": 1.0, "max_bio_length": 300, "type": "bio"}'

```
Alternatively, you can use *_test_curl.txt files provided in tests/files folder.

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

## Acknowledgements

This project utilizes [AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) for anime-style image transformation.

- **AnimeGAN2-PyTorch** by [bryandlee](https://github.com/bryandlee)  
  - Repository: [AnimeGAN2-PyTorch](https://github.com/bryandlee/animegan2-pytorch)
  - License: [MIT](https://opensource.org/licenses/MIT).

----------

## **🔗 Links**
🔹 **Live Demo**: [ficbotweb.com](https://ficbotweb.com)  
🔹 **Ficbot**: [ficbot](https://github.com/Pythonimous/ficbot)  
🔹 **Dataset**: [Kaggle](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [Kaggle](https://www.kaggle.com/code/ophelion/myanimelist-dataset-exploratory-notebook)  
