# 🧠💊 Recommend-Medicine: An AI-Powered Drug Suggestion Platform

Welcome to **Recommend-Medicine**, a smart and scalable healthcare solution powered by **Machine Learning** 🤖 and **Python** 🐍. This project uses user-reported symptoms to intelligently recommend appropriate medicines 💊. Whether you're a developer, data scientist, or a healthcare tech enthusiast, this project offers an engaging intersection of **AI**, **healthcare**, and **web development**.

---

## 📚 Table of Contents

- [🌟 Features](#-features)
- [📦 Tech Stack](#-tech-stack)
- [🧪 Machine Learning Pipeline](#-machine-learning-pipeline)
- [🖥️ Web Application Overview](#-web-application-overview)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [⚙️ How It Works](#-how-it-works)
- [🧠 Model Training](#-model-training)
- [📊 Data & Preprocessing](#-data--preprocessing)
- [💡 Potential Improvements](#-potential-improvements)
- [🤝 Contribution Guide](#-contribution-guide)
- [📜 License](#-license)
- [📬 Contact](#-contact)

---

## 🌟 Features

The **Recommend-Medicine** system offers an integrated ML model that maps symptoms to the most appropriate medication. It is designed for both educational and demonstration purposes in healthcare tech. Here are some key features:

- 🔍 Accepts user symptoms and returns suitable medicine suggestions.
- 🤖 Leverages a machine learning model trained on curated datasets.
- 🌐 A Flask-based web interface for smooth user interaction.
- 💾 Includes serialized `.pkl` file for quick model reusability.
- 📊 Uses two datasets for improved diversity and training potential.
- 🧪 Built-in scripts for model training and predictions.

---

## 📦 Tech Stack

| 🔧 Technology | 💡 Purpose |
|--------------|------------|
| Python 🐍 | Core programming language |
| Flask 🌐 | Web framework for the frontend/backend |
| Pandas 📊 | Data manipulation and loading |
| Scikit-learn 🤖 | Machine learning algorithms and evaluation |
| Pickle 💾 | Saving and loading trained models |
| HTML/CSS 🎨 | Web UI templates |
| CSV Files 📁 | Datasets containing symptoms and medicines |

---

## 🧪 Machine Learning Pipeline

The pipeline begins with structured data ingestion, processes it for training, and builds a classifier model which can later be used to make real-time predictions via the web interface. Here's how it works:

1. 📥 **Data Loading**: Reads CSV files into dataframes.
2. 🧼 **Preprocessing**: Handles missing values, duplicates, and normalizes symptoms.
3. 🧠 **Model Training**: A classifier (e.g., Decision Tree, RandomForest) is trained.
4. 📈 **Evaluation**: Accuracy and performance are measured on test data.
5. 💾 **Serialization**: The trained model is saved as `model.pkl`.
6. 🔮 **Inference**: Predictions are made on user input in real-time via Flask.

---

## 🖥️ Web Application Overview

The user-facing web interface is powered by Flask. It features a simple form where users can enter their symptoms.

### 🔍 User Journey:

1. The user accesses the web app via `localhost:5000`.
2. They input their symptoms into the web form.
3. They click the "Predict" button.
4. The backend processes the input and returns a medicine recommendation.

---

## 📁 Project Structure

```bash
Recommend-Medicine/
├── .vscode/               # Editor settings
├── templates/             # HTML templates for web UI
│   └── index.html         # Main interface page
├── app.py                 # Flask application logic
├── train_model.py         # Machine Learning model training script
├── model.pkl              # Pre-trained serialized model
├── requirements.txt       # Python dependencies
├── Data.csv               # Primary symptom-to-medicine dataset
├── symbipredict_2022.csv  # Additional dataset (for future enhancement)
└── venv/                  # (Optional) Python virtual environment
```

---

## 🚀 Getting Started

To run this project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/GHVAcTive/Recommend-Medicine.git
cd Recommend-Medicine
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Launch the Application

```bash
python app.py
```

✅ Open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## ⚙️ How It Works

Once the user inputs their symptoms on the web form:

- The symptoms are passed to the backend.
- The model loads from `model.pkl`.
- A prediction is made based on the input.
- The recommended medicine is displayed on the result page.

This real-time system bridges user-friendly web interaction with smart ML prediction.

---

## 🧠 Model Training

The training script is located in `train_model.py`. It includes:

- Loading data from `Data.csv`.
- Preprocessing the input features and target labels.
- Training a classifier using scikit-learn.
- Saving the model as `model.pkl`.

To retrain:

1. Add more data to the CSV files.
2. Adjust the model parameters in `train_model.py`.
3. Rerun the script to update the model.

---

## 📊 Data & Preprocessing

Two CSV files are included:

- `Data.csv`: Primary dataset containing mappings from symptoms to medicines.
- `symbipredict_2022.csv`: Additional dataset for extended use.

### 🔧 Preprocessing Includes:

- Dropping missing or invalid rows.
- Converting all symptoms to lowercase.
- Encoding categorical features.

You can enrich the dataset with external symptom-to-drug mappings to improve model performance.

---

## 💡 Potential Improvements

This project is an excellent starting point, but here are a few suggestions to enhance its capabilities:

- 🧬 Use NLP to parse symptoms described in natural language.
- 📈 Add dashboard analytics using Plotly or Dash.
- 🌍 Deploy the app using Heroku, Render, or AWS Lambda.
- 🧪 Integrate online APIs for dynamic drug lookups.
- 🤝 Implement user accounts for personalized recommendations.
- 🚀 Switch to FastAPI for high-performance REST APIs.

---

## 🤝 Contribution Guide

We love contributions from the community! Here's how you can help:

1. 🍴 Fork the repository.
2. 🌿 Create your feature branch (`git checkout -b feature/fooBar`).
3. 💾 Commit your changes (`git commit -am 'Add some fooBar'`).
4. 🚀 Push to the branch (`git push origin feature/fooBar`).
5. 📬 Create a new Pull Request.

All suggestions, improvements, or bug reports are welcome!

---

## 📜 License

This project is licensed under the MIT License. That means it's free to use, modify, and distribute in personal or commercial projects.

---

## 📬 Contact

Have questions or want to collaborate? Reach out!

- GitHub: [GHVAcTive](https://github.com/GHVAcTive)
- Email: 📧 _your.email@example.com_

---

🌟 Thank you for checking out **Recommend-Medicine**. We hope it inspires smarter and more accessible healthcare applications!

