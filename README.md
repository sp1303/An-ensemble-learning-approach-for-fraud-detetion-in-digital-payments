# An Ensemble Learning Approach for Fraud Detection in Digital Payments

## 📌 Project Overview

Digital payment systems have made financial transactions faster and more convenient, but they have also increased the risk of fraudulent activities. Detecting fraudulent transactions accurately is an important challenge because fraud can lead to financial losses and reduced customer trust.

This project explores an **ensemble learning approach for fraud detection in digital payments** using multiple machine learning algorithms. It compares individual classification models with a stacking ensemble model to classify transactions as fraudulent or legitimate.

The project also includes a Flask-based web application that allows transaction features to be submitted for prediction.

## 🎯 Objectives

- Detect potentially fraudulent digital payment transactions using machine learning.
- Implement and compare multiple classification algorithms.
- Apply data preprocessing and feature scaling.
- Explore SMOTE for handling class imbalance.
- Use stacking ensemble learning to combine predictions from multiple base models.
- Develop a web application for transaction classification.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and model evaluation |
| Imbalanced-learn | SMOTE-based resampling |
| Matplotlib | Visualization and model comparison |
| Joblib | Saving and loading trained models |
| Flask | Web application and prediction API |
| HTML, CSS, JavaScript | Web interface |

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of a transaction belonging to the fraudulent class. It serves as a baseline model for binary classification.

### 2. Random Forest

Random Forest combines predictions from multiple decision trees. It can learn nonlinear relationships and interactions between transaction features.

### 3. K-Nearest Neighbors (KNN)

KNN classifies a transaction based on nearby examples in the feature space. Feature scaling is important because the algorithm uses distances between data points.

### 4. Stacking Ensemble Learning

The project uses a stacking approach with Logistic Regression, Random Forest, and KNN as base estimators and Logistic Regression as the final estimator.

The final estimator learns from the predictions of the base models to generate a combined classification.

## 🔄 Project Workflow

1. **Dataset preparation:** Load transaction data and identify the target variable.
2. **Data preprocessing:** Separate input features and target labels.
3. **Train-test split:** Divide the dataset into training and testing subsets.
4. **Class balancing:** Apply SMOTE to the training data where configured.
5. **Feature scaling:** Standardize numerical features using `StandardScaler`.
6. **Model training:** Train individual machine learning models.
7. **Ensemble training:** Train a stacking classifier using the base estimators.
8. **Model evaluation:** Compare model predictions using appropriate classification metrics.
9. **Model persistence:** Save trained models and preprocessing objects using Joblib.
10. **Web application:** Use Flask to accept transaction features and return predictions.

## 📂 Project Structure

```text
An-ensemble-learning-approach-for-fraud-detetion-in-digital-payments/
│
├── app.py
├── train_models.py
├── lr.py
├── rf.py
├── knn.py
├── min.py
├── generate_dataset.py
├── rename_features.py
├── requirements.txt
├── accurate_ensemble_plot.png
├── comparison_plot.png
│
├── static/
│   └── CSS, JavaScript and other static assets
│
└── templates/
    └── HTML templates
```

*Note: This structure summarizes the main files. Generated datasets, trained model files, and other assets may also be required to run the complete application.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.9 or a compatible version for the installed dependencies
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/sp1303/An-ensemble-learning-approach-for-fraud-detetion-in-digital-payments.git
```

### Step 2: Navigate to the project directory

```bash
cd An-ensemble-learning-approach-for-fraud-detetion-in-digital-payments
```

### Step 3: Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Step 1: Generate the dataset

If using the included dataset-generation script, run:

```bash
python generate_dataset.py
```

### Step 2: Train the machine learning models

```bash
python train_models.py
```

### Step 3: Start the Flask application

```bash
python app.py
```

Open the local URL displayed in the terminal in your browser.

**Important:** The dataset filename and feature names must match what the training and application scripts expect. Verify these settings before running the workflow. The scripts may require configuration changes to work together correctly.

## 📊 Model Evaluation

Fraud detection should be evaluated using multiple classification metrics rather than accuracy alone.

- **Accuracy:** Percentage of correctly classified transactions.
- **Precision:** Percentage of predicted fraudulent transactions that are actually fraudulent.
- **Recall:** Percentage of actual fraudulent transactions detected by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives.

Recall is especially important when undetected fraud is costly, while precision helps control false fraud alerts. The appropriate balance depends on the application.

Model performance should be calculated from actual predictions on a held-out test set. Results should be added to this README only after the evaluation has been verified.

## 🌐 Application Features

- Flask-based prediction endpoint.
- Support for individual model predictions and stacking-based predictions.
- Transaction classification into fraudulent or legitimate categories.
- Probability-based prediction output.
- JSON responses from the prediction API.

The exact features available through the web interface depend on the current implementation.

## 🔮 Future Enhancements

- Evaluate models on a verified real-world or appropriately sourced transaction dataset.
- Improve the consistency of the training and inference pipelines.
- Add cross-validation and leakage-safe stacking evaluation.
- Tune classification thresholds based on precision-recall trade-offs.
- Add explainability techniques to understand important fraud indicators.
- Improve the user interface with transaction summaries and model evaluation dashboards.
- Deploy the application in a secure environment with appropriate input validation and monitoring.

## ⚠️ Limitations

- Synthetic datasets may not represent the complexity of real payment fraud.
- Model performance depends on data quality, feature selection, and evaluation methodology.
- False positives can incorrectly flag legitimate transactions, while false negatives can allow fraud to go undetected.
- Predictions are estimates and should not be treated as definitive proof of fraud.
- The current code and dataset configuration should be validated before production use.

## 🔐 Disclaimer

This project is intended for educational and research purposes. It is not a production-ready financial fraud detection system. Do not use it to make real financial decisions without proper validation, security controls, and domain-specific testing.

## 👩‍💻 Author

**Shatakshi Pandiya**

- GitHub: [sp1303](https://github.com/sp1303)
- LinkedIn: [Shatakshi Pandiya](https://www.linkedin.com/in/shatakshi-pandiya-5bb4513a6/)

---

⭐ If you find this project useful, consider giving the repository a star.
