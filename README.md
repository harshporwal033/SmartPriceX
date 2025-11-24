# 💰 SmartPriceX – AI-Driven Multimodal Product Price Prediction System

SmartPriceX is a multimodal machine-learning system designed to **predict optimal e-commerce product prices** by combining image embeddings, textual metadata, and tabular features. The pipeline integrates deep-learning–based image feature extraction, transformer-based text representations, and regression optimized for SMAPE to deliver highly accurate pricing predictions.

---

## 🔍 Overview

Accurate product pricing is a key challenge for online retailers. SmartPriceX addresses this by analysing multiple modalities of data: product images, titles/descriptions, item pack quantities (IPQ) and unit attributes. By fusing visual, textual and tabular signals, the system learns what factors influence pricing and predicts real-world prices accordingly.

---

## 🌟 Key Features

### Multimodal Feature Engineering

* **Image pipeline**: Uses an ensemble of ResNet-18, DenseNet-121 and EfficientNet-B0 models (via PyTorch and timm) to generate 512-dimensional embeddings.
* **Text pipeline**: Uses the `all-MiniLM-L6-v2` Sentence Transformer model to create dense text embeddings from product titles and descriptions.
* **Tabular features**: Extracts IPQ values, unit types via regex parsing and one-hot encoding, and combines them with text embeddings.

### Model & Pipeline

* Data pipeline supports image download, embedding generation, and feature concatenation.
* Final regression model (custom or ensemble) optimised for **SMAPE** (Symmetric Mean Absolute Percentage Error).
* Output format standardised for submission: predictions in `test_out.csv` with `sample_id` and `price`.

### Implementation Highlights

* Notebook-driven workflow: easy to follow and reproduce.
* Efficient execution with multiprocessing, GPU support, and safe file operations.
* Modular codebase for switching out components or extending with new modalities.

---

## 📂 Project Structure

```
SmartPriceX/
├── AMLC/                             # Root folder for pipeline  
│   ├── download_images.ipynb         # Notebook for image download  
│   ├── utils.py                      # Helper functions (download logic, retries)  
│   ├── image_embedding.ipynb         # Notebook to generate image embeddings  
│   ├── text_embedding.ipynb          # Notebook to generate text + tabular embeddings  
│   └── AMLC_2025/                    # Main model folder  
│       ├── data/                     # Raw CSVs + intermediate feature files  
│       │   ├── train.csv  
│       │   ├── test.csv  
│       │   ├── test_ids.csv  
│       │   ├── X_train_img_ensemble.npy  
│       │   ├── X_test_img_ensemble.npy  
│       │   ├── X_train_text_features_bert.npz  
│       │   ├── X_test_text_features_bert.npz  
│       │   └── y_train_full.npy  
│       └── Main.ipynb                # Notebook that combines features, trains the model & generates submission  
├── README.md                         # Project documentation  
└── requirements.txt                  # Python dependencies  
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/harshporwal033/SmartPriceX.git
cd SmartPriceX
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Data setup

Ensure you place the dataset files (`train.csv`, `test.csv`, etc.) inside the `AMLC/AMLC_2025/data/` directory.

### 5. Run the workflow

* Download images: `jupyter notebook AMLC/download_images.ipynb`
* Generate image embeddings: `jupyter notebook AMLC/image_embedding.ipynb`
* Generate text & tabular features: `jupyter notebook AMLC/text_embedding.ipynb`
* Train model & make predictions: `jupyter notebook AMLC/AMLC_2025/Main.ipynb`

---

## 📄 Output Format

After successful execution, the pipeline will generate `test_out.csv` in the project root (or specified output folder) with the following format:

| Column    | Description                                    |
| --------- | ---------------------------------------------- |
| sample_id | Unique identifier matching each test record    |
| price     | Predicted product price (positive float value) |

Example entry:

```
75001,14.99
```

---

## 🧰 Technologies & Tools Used

* **Programming Language:** Python
* **Libraries:** NumPy, Pandas, Scikit-learn, Sentence-Transformers, PyTorch, timm
* **Notebooks:** Jupyter
* **Feature Engineering:** Image embeddings, text embedding, regex parsing, one-hot encoding
* **Modeling:** Regression algorithm optimised for SMAPE metric
* **Development Tools:** Virtual environments, modular notebooks, clear folder structure

---

## 👨‍💻 Author

Project developed by **Harsh Porwal**.
(You may add your contributions or co-author names as needed.)

---

## 🤝 Contributions

Contributions are welcome! If you’d like to improve feature extraction, try a different architecture, or optimise the model further:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your‐idea`
3. Commit your changes: `git commit -m "Feat: improved …"`
4. Push to your branch: `git push origin feature/your‐idea`
5. Open a Pull Request with details of your improvement

---

