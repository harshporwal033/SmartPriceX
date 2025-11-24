💰 Smart Product Pricing (ML Challenge 2025 Submission)
<!-- A concise, one or two-sentence summary of what your project does. -->

This repository presents a multimodal machine learning solution for the Smart Product Pricing Challenge. The goal is to predict the optimal price of e-commerce products by holistically analyzing both their textual attributes (title, description, IPQ from catalog_content) and visual features extracted from product images. The model is optimized to minimize the Symmetric Mean Absolute Percentage Error (SMAPE).

📋 Table of Contents
Project Structure

Features

Installation

Running Predictions

Output Format

Contributing

License

📂 Project Structure
The project is structured into a root directory (AMLC) containing utility scripts and notebooks for feature extraction, and a subdirectory (AMLC_2025) for the core data and final model training.

AMLC/
├── download_images.ipynb       # Notebook for downloading all product images using utils.py
├── utils.py                    # Helper functions, including the image download retry logic
├── image_embedding.ipynb       # Notebook for generating image features (ResNet, DenseNet, EfficientNet ensemble)
├── text_embedding.ipynb        # Notebook for generating text features (Sentence-Transformer, regex parsing)
├── AMLC_2025/
│   ├── data/                   # Directory to store raw CSVs, intermediate features (.npy, .npz), and the target variable
│   │   ├── train.csv           # Provided training data with price labels
│   │   ├── test.csv            # Provided test data for prediction
│   │   ├── test_ids.csv        # Extracted sample_id list for submission formatting
│   │   ├── X_train_img_ensemble.npy   # Saved 512D image embeddings for training data
│   │   ├── X_test_img_ensemble.npy    # Saved 512D image embeddings for test data
│   │   ├── X_train_text_features_bert.npz  # Combined text embeddings, IPQ value, and unit features (training)
│   │   ├── X_test_text_features_bert.npz   # Combined text embeddings, IPQ value, and unit features (test)
│   │   └── y_train_full.npy    # Saved array of price (target) values from train.csv
│   └── Main.ipynb              # Final notebook for combining features, training the multimodal model, and generating test_out.csv
└── README.md  


✨ Features
Enhanced Multimodal Feature Extraction: * Image: Leverages an ensemble of ResNet-18, DenseNet-121, and EfficientNet-B0 (using Soft Voting to a 512-dimension vector) for robust image feature generation.

Text: Uses the all-MiniLM-L6-v2 Sentence Transformer to create dense embeddings from the combined product name and description.

Comprehensive Feature Engineering: The final text feature set combines the dense text embeddings with tabular features extracted via regex parsing (Item Pack Quantity and one-hot encoded Unit).

Optimized Image Pipeline: Utilizes PyTorch, timm, and optimized multiprocessing/CUDA settings for fast and efficient image feature extraction on large datasets.

Optimized for SMAPE: Regression architecture specifically tuned for high performance against the challenge's core evaluation metric.

Automated Data Pipeline: Includes routines for safe, throttled image downloading using utils.py.

Standardized Output: Generates prediction file (test_out.csv) in the exact format required for submission.

🛠️ Installation
Follow these steps to set up the project environment locally.

Prerequisites
Ensure you have the following installed on your system:

Python 3.9+

Git

Jupyter Notebook (or JupyterLab) is required to execute the .ipynb files.

PyTorch (Installation with CUDA support is highly recommended for the image embedding step)

Key Python Libraries (managed via requirements.txt):

pandas, timm, torchvision, numpy,

sentence-transformers, scikit-learn, scipy.

Steps
Clone the repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

Set up virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:
Ensure your requirements.txt file lists all necessary libraries, including PyTorch.

pip install -r requirements.txt

Data Setup:
Ensure that the provided train.csv and test.csv files are placed inside the AMLC_2025/data/ directory.

🏃 Running Predictions
The workflow involves executing four main steps sequentially via their respective Jupyter Notebooks.

Download Images:
Execute the download notebook to fetch all images from the provided URLs into the image folders:

# Open and run all cells in the notebook:
jupyter notebook download_images.ipynb

Generate Image Embeddings:
Run the dedicated notebook to process and save the multimodal image features (X_train_img_ensemble.npy and X_test_img_ensemble.npy) in the AMLC_2025/data/ folder:

# Open and run all cells in the notebook:
jupyter notebook image_embedding.ipynb

Note: This step is GPU-optimized and will be significantly faster if CUDA is available.

Generate Text Features:
Run the dedicated notebook to parse the catalog_content, generate text embeddings, and combine them with engineered tabular features (X_train_text_features_bert.npz and X_test_text_features_bert.npz):

# Open and run all cells in the notebook:
jupyter notebook text_embedding.ipynb

Generate Final Submission:
Run the primary notebook which handles loading all combined features, training the final multimodal model, inference on the test set, and outputting the final submission file (test_out.csv):

# Navigate to the subdirectory and run the final model notebook:
cd AMLC_2025
jupyter notebook Main.ipynb

📄 Output Format
The successful execution of the final script will generate a file named test_out.csv in the root directory, formatted exactly like dataset/sample_test_out.csv.

Column Name

Description

Example

sample_id

Unique identifier matching the test records.

75001

price

Predicted product price (positive float value).

14.99

🤝 Contributing
We welcome contributions! If you have suggestions or want to improve the model's performance, please follow these steps:

Fork the project.

Create your feature branch (git checkout -b feature/model-v2).

Commit your changes (git commit -m 'Feat: Improved feature extraction for images').

Push to the branch (git push origin feature/model-v2).

Open a Pull Request detailing your changes and performance improvements.

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
(Note: The final model must be MIT/Apache 2.0 Licensed and up to 8 Billion parameters as per challenge rules.)
