# 📝 Duplicate Question Detection

## 📌 Project Overview
This project builds a **machine learning pipeline** to detect whether two questions are duplicates. It is a **binary classification problem**, where the output is:
- `1` → Duplicate questions
- `0` → Non-duplicate questions

The project follows a structured ML workflow, starting from data exploration to building a stacked ensemble model for best performance.

---

## 🚀 Steps in the Pipeline

### 1️⃣ Exploratory Data Analysis (EDA)
- Inspected dataset structure, missing values, and distributions.
- Analyzed question lengths, duplicate ratios, and common patterns.

### 2️⃣ Data Preprocessing
- Removed **stopwords** using spaCy.
- Removed **HTML tags** with BeautifulSoup.
- Expanded **contractions** (e.g., `can't → cannot`).
- Lowercased and cleaned text for consistency.

### 3️⃣ Feature Engineering
- Generated features such as:
  - **Text similarity scores** (cosine similarity with TF-IDF).
  - **Length-based features** (word/char count differences).
  - **Word overlap ratios**.
- Extracted **Sentence Transformer embeddings**:
  - `all-MiniLM-L6-v2`
  - `all-MPNet-base-v2`

### 4️⃣ Model Training
Implemented and trained multiple models:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Neural Network (MLP)

### 5️⃣ Hyperparameter Tuning
- Used **40% of the dataset** for tuning.
- Applied `GridSearchCV` / `RandomizedSearchCV`.
- Selected best hyperparameters for each model.

### 6️⃣ Final Model Training
- Trained final models on the **entire dataset** with best parameters.
- Saved individual models (`.pkl` format).

### 7️⃣ Stacking Ensemble
- Selected the **best 3 models** based on validation performance.
- Built a **StackingClassifier** with Logistic Regression as meta-learner.
- Used **cross-validation (cv=5)** to avoid data leakage.
- Saved final stacked model as **`stacked_final_model.pkl`**.

---

## 📂 Project Structure
```
├── data/                  # Raw and processed data
├── notebooks/             # EDA and experiments
├── models/                # Saved trained models (.pkl)
├── main.py                # Main training pipeline
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚡ Requirements
- Python 3.8+
- scikit-learn
- xgboost
- spaCy
- sentence-transformers
- numpy, pandas
- joblib
- matplotlib, seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage
### Training the Model
```bash
python main.py
```

### Loading and Using Final Model
```python
import joblib

# Load stacked model
model = joblib.load("stacked_final_model.pkl")

# Example prediction
q1 = "How can I learn machine learning?"
q2 = "What is the best way to study ML?"
X_new = preprocess_and_vectorize(q1, q2)  # custom preprocessing function
y_pred = model.predict([X_new])
print("Duplicate" if y_pred[0] == 1 else "Not Duplicate")
```

---

## 📊 Results
- Achieved strong performance with stacked ensemble.
- Ensemble outperformed individual models in terms of **F1-score** and **AUC**.

---

## ✨ Future Improvements
- Experiment with **BERT fine-tuning**.
- Add more linguistic features (e.g., dependency parsing).
- Deploy as an API using Flask/FastAPI.

---

## 👨‍💻 Author
Developed by **Priyanshu Kashyap**

