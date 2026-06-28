# ⚙️ ML Training Pipeline Dashboard

ML Training Pipeline Dashboard is a personal AI project that provides an interactive web interface to automate and visualize the end-to-end Machine Learning workflow.

Users can upload raw datasets, configure preprocessing steps, define hyperparameters, train classification models, and evaluate performance metrics dynamically.

This project was built as a portfolio project for AI Engineer / Machine Learning Engineer internship applications.

---

## ✨ Demo Features

- Upload custom `.csv` datasets for training
- Dynamic selection of Target (y) and Feature (X) columns
- Handle missing values (Mean, Median, Most Frequent imputation)
- Apply categorical encoding (One-Hot Encoding)
- Apply feature scaling (StandardScaler, MinMaxScaler)
- Configure Train/Test data split ratio and random state
- Train machine learning models (Random Forest, Logistic Regression)
- Tune model hyperparameters interactively via UI sliders
- Generate classification metrics (Accuracy, Precision, Recall, F1-Score)
- Visualize Confusion Matrix for model evaluation
- Strict prevention of Data Leakage by isolating train/test transformations
- Maintain application state securely across pipeline steps

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Streamlit | Web UI framework |
| Scikit-learn | ML models & preprocessing |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Visualization |

---

## 🏗️ Project Architecture

```
User uploads CSV dataset
        ↓
Store raw data in Session State
        ↓
User selects Target and Features
        ↓
Configure Imputation, Encoding, and Scaling strategies
        ↓
Split data into Train and Test sets
        ↓
Apply preprocessing transformations strictly (fit on Train, transform on Test)
        ↓
Train Model (Random Forest / Logistic Regression) with custom hyperparameters
        ↓
Perform predictions on the Test set
        ↓
Calculate performance metrics
        ↓
Render Metrics and Confusion Matrix in UI
```

---

## 📁 Folder Structure

```
ml-pipeline-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── data/
│   └── sample_dataset.csv
│
└── assets/
    └── architecture_diagram.png
```

---

## 🧩 Core Modules

### `data_ingestion.py`
Handles file upload mechanisms, reads CSV data using Pandas, implements caching for performance optimization, and displays a data preview.

### `preprocessing.py`
Manages user configurations for handling missing data, scaling numerical features, and encoding categorical variables. Saves transformation parameters without prematurely applying them to prevent data leakage.

### `model_training.py`
Executes the train-test split, systematically applies the preprocessing configurations to isolate the train and test sets, initializes the chosen Scikit-learn model, and performs the training loop.

### `evaluation.py`
Extracts the trained model and processed test data to generate predictions. Calculates core classification metrics and plots the Confusion Matrix using Matplotlib for visual analysis.

### `app.py`
Main Streamlit application that orchestrates the UI layout, sidebar navigation, and manages the session state to ensure seamless transitions between pipeline stages.

---

## 🤖 Machine Learning Techniques Demonstrated

This project demonstrates practical expertise in core Data Science and Machine Learning principles:

- **Exploratory Data Analysis (EDA):** Dynamic feature selection and data parsing.
- **Feature Engineering:** Imputation, One-Hot Encoding, Min-Max/Standard Scaling.
- **Model Validation:** Strict Train/Test splitting.
- **Algorithm Implementation:** Ensemble methods (Random Forest) and Linear models (Logistic Regression).
- **Performance Evaluation:** Multi-class metric extraction and graphical confusion matrix representation.
- **Software Engineering:** Modularity, state management, and separation of concerns.

---

## 📋 Example Input Configuration

```
Dataset: Telco Customer Churn
Target Column: Churn
Feature Columns: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod
Missing Value Strategy: Median
Scaling Strategy: StandardScaler
Test Size Ratio: 0.2
Model: Random Forest
n_estimators: 100
max_depth: 10
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/DungDao23092005/ml-pipeline-dashboard.git
cd ml-pipeline-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows CMD:**
```bash
venv\Scripts\activate
```

**Windows PowerShell:**
```bash
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📝 Development Process

- [feat: build model training pipeline with scikit-learn](https://github.com/DungDao23092005/ml-training-pipeline-dashboard/commit/08213d3a12062d104c723c3360b8dfdb8b4cd151)
- [feat: add data preprocessing logic](https://github.com/DungDao23092005/ml-training-pipeline-dashboard/commit/b8faec1ba0a487f586673b97e473516e99875923)
- [feat: implement data ingestion and caching module](https://github.com/DungDao23092005/ml-training-pipeline-dashboard/commit/aa9e1841c4c5eb2cf12903fa6250b7a1aa23f4e9)
- [chore: add requirements.txt for dependencies](https://github.com/DungDao23092005/ml-training-pipeline-dashboard/commit/990058a4a701d187acf563db508afd22fc565f07)

---

## ✅ Current Status

This project has completed the main MVP features:

- [x] Streamlit multi-step UI
- [x] File upload and caching
- [x] Dynamic data preprocessing
- [x] Supervised learning model training
- [x] Hyperparameter tuning interface
- [x] Real-time metrics calculation
- [x] Matplotlib visual integration

---

## 🔮 Future Improvements

- Add support for Regression algorithms (Linear Regression, XGBoost Regressor)
- Add Unsupervised Learning capabilities (K-Means Clustering)
- Implement feature importance visualizations
- Add downloadable reports (PDF/Markdown) of the model evaluation results
- Containerize the application using Docker
- Deploy to cloud platforms like Azure or AWS

---

## 👤 Author

**Dung Dao**

- GitHub: [@DungDao23092005](https://github.com/DungDao23092005)

---

## 📄 License

This project is intended for educational and portfolio purposes.