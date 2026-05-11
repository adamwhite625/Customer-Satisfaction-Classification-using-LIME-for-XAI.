# Customer Satisfaction Classification using LIME for XAI

## 📋 Project Overview

This project aims to **classify customer satisfaction levels** from E-Commerce data using machine learning models. Furthermore, the project focuses on **Explainable AI (XAI)** by using **LIME** (Local Interpretable Model-Agnostic Explanations) to explain model decisions, helping understand why the model makes specific predictions.

### 🎯 Main Objectives:

- Build a model to predict customer satisfaction (binary classification: Satisfied / Dissatisfied)
- Compare the performance of different machine learning algorithms
- Provide clear explanations for model predictions using LIME
- Evaluate model quality and reliability

---

## 📊 Dataset

### Data Source:

- **Dataset**: `E-Commerce Customer Data.csv`
- **Size**: ~12,730 customer records
- **Format**: CSV

### Main Data Fields:

| Field                   | Data Type      | Description                                      |
| ----------------------- | -------------- | ------------------------------------------------ |
| `Customer_ID`           | String         | Customer identifier                              |
| `Customer_Satisfaction` | Integer (1-10) | **Target variable**: Satisfaction level 1-10     |
| `Age`                   | Integer        | Customer age                                     |
| `Purchase_Amount`       | Float          | Purchase amount by customer                      |
| `Return_Rate`           | Float          | Product return rate                              |
| `Delivery_Status`       | String         | Delivery status (Delivered, Returned, Pending)   |
| `Gender`                | String         | Gender (Male/Female)                             |
| `Purchase_Channel`      | String         | Purchase channel (Website, Mobile App, In-store) |
| `Discount_Used`         | Boolean        | Whether discount code was used                   |
| `Time_of_Purchase`      | DateTime       | Purchase time                                    |
| `Time_to_Decision`      | Integer        | Time needed for customer to decide on purchase   |

---

## 🔍 Data Processing Methods

### 1. Preprocessing:

- **Handling Missing Values**: Using `median` for numerical variables, `mode` for categorical variables
- **Feature Engineering Techniques**:
  - Create temporal features: `Purchase_Month`, `Purchase_Day_of_Week`, `Purchase_Hour`, `Is_Weekend`
  - Standardize numerical variables using `StandardScaler`
  - One-Hot encoding for categorical variables

### 2. Two Different Approaches:

#### **Strategy 1: Baseline (Threshold 5/6)**

- **Objective**: Classification based on 5/6 threshold
  - Class 0: Satisfaction level ≤ 5
  - Class 1: Satisfaction level > 5
- **Data**: Use entire dataset (~12,730 samples)
- **Advantages**: Uses all data, no information loss
- **Disadvantages**: "Noise zone" data (4-5-6) makes model decision difficult

#### **Strategy 2: Filtered (Remove 4-5-6)**

- **Objective**: Classification with only clear data
  - Class 0: Satisfaction level 1-3 (Very dissatisfied)
  - Class 1: Satisfaction level 8-10 (Very satisfied)
  - **Removed**: Data with satisfaction level 4-5-6-7 (noise zone)
- **Data**: ~75% of dataset (~9,500 samples) after filtering
- **Advantages**: Clearer dataset, better model learning
- **Disadvantages**: Loss of part of data

---

## 🤖 Machine Learning Models

The project compares **10+ different algorithms**:

### Traditional Models (Shallow Learning):

1. **Logistic Regression** - Basic linear model
2. **Decision Tree** - Decision tree classifier
3. **Random Forest** - Ensemble of decision trees
4. **Support Vector Machine (SVM)** - Support vector classifier
5. **Naive Bayes** - Probabilistic classifier
6. **AdaBoost** - Boosting classifier
7. **Gradient Boosting** - Gradient boosting classifier
8. **LightGBM** - Light Gradient Boosting Machine (Baseline)
9. **K-Nearest Neighbors (KNN)** - k-NN classifier

### Deep Learning Models:

1. **MLP (Multi-Layer Perceptron)** - Multi-layer neural network
2. **1D CNN** - 1-Dimensional convolutional neural network
3. **MLP VIP Best** - Optimized version of MLP

---

## 🔬 Results and Evaluation

### Evaluation Metrics:

- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of true positive predictions
- **Recall**: Proportion of satisfied customers detected
- **F1-Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Displays correct/incorrect predictions

### Baseline Results:

- **Baseline LightGBM (Threshold 5/6)**: ~71.33% Accuracy
- **Filtered Approach (1-3 vs 8-10)**: Higher accuracy due to clearer data

---

## 🧠 Explainable AI (XAI) with LIME

### What is LIME?

**LIME** (Local Interpretable Model-Agnostic Explanations) is a technique to explain machine learning model predictions by:

1. Taking a specific prediction from the model
2. Creating small variations of the input data
3. Observing how the model behaves with these variations
4. Identifying the most important features affecting the prediction

### Benefits:

✅ Explain model predictions in a human-understandable way
✅ Detect bias or unexpected model behavior
✅ Increase confidence when deploying models in production
✅ Evaluate models not just by accuracy but also by reasoning logic

### Usage Example:

```python
# Explain prediction for a specific customer
exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
exp.show_in_notebook()  # Show important features
```

---

## 📁 Directory Structure

```
doAn/
├── README.md                          # This documentation
├── dataset/
│   └── E-Commerce Customer Data.csv   # Raw data
├── models/                            # Directory containing saved models
│   ├── model_baseline_LGBM_5_6.joblib
│   ├── model_logistic_regression.joblib
│   ├── model_decision_tree.joblib
│   ├── model_random_forest.joblib
│   ├── model_svm.joblib
│   ├── model_naive_bayes.joblib
│   ├── model_adaboost.joblib
│   ├── model_gradient_boosting.joblib
│   ├── model_lightgbm.joblib
│   ├── model_mlp.h5                   # Deep Learning Model
│   ├── model_mlp_VIP_Best.h5
│   └── model_1d_cnn.h5
├── source/                            # Source code
│   ├── finalNotebook.ipynb            # Jupyter Notebook (.ipynb format)
│   ├── finalNotebook.html             # HTML version of Notebook
│   └── finalnotebook.py               # Python version of Notebook
└── report/                            # Reports directory (if available)
```

---

## 🚀 How to Run the Project

### Requirements:

- Python 3.7+
- Jupyter Notebook (or Google Colab)

### Required Libraries:

```bash
pip install pandas numpy scikit-learn tensorflow keras lightgbm lime matplotlib seaborn
```

### Running Steps:

#### **Option 1: Run on Jupyter Notebook**

```bash
jupyter notebook source/finalNotebook.ipynb
```

#### **Option 2: Run on Google Colab**

1. Upload `finalNotebook.ipynb` to Google Colab
2. Connect Google Drive to save/load data
3. Run cells in sequence

#### **Option 3: Run Python File**

```bash
python source/finalnotebook.py
```

### Notebook Structure:

| Step       | Content                             |
| ---------- | ----------------------------------- |
| **Step 1** | Environment setup & Load libraries  |
| **Step 2** | Load and Explore Data (EDA)         |
| **Step 3** | Baseline Experiment (Threshold 5/6) |
| **Step 4** | Filtered Experiment (Remove 4-5-6)  |
| **Step 5** | Compare & Explain (XAI) with LIME   |

---

## 📈 Key Results

### Data Analysis (EDA):

- **Satisfaction Level Distribution**: From 1-10 with some peaks
- **"Noise Zone" (4-7 points)**: Large number of customers with neutral satisfaction
- **Return Rate**: Strong correlation with dissatisfaction
- **Delivery Status**: Major impact on satisfaction level
  - "Delivered" → High satisfaction
  - "Returned" → Dissatisfaction
  - "Pending" → Unclear

### Model Performance:

- **LightGBM** (Baseline): 71.33% Accuracy ✓
- **Other models**: Detailed comparison results in notebook

### LIME Insights:

- Most important features: `Return_Rate`, `Delivery_Status`, `Purchase_Amount`
- Model decisions based on reasonable factors
- No major bias detected

---

## 💡 Future Development Directions

1. **Data Improvement**:
   - Collect more data to increase dataset size
   - Thoroughly clean and validate data quality

2. **Advanced Models**:
   - Use Grid Search / Random Search for hyperparameter optimization
   - Try new models: XGBoost, CatBoost, Voting Classifier
   - Apply more complex Ensemble methods

3. **Extended XAI**:
   - Use SHAP (SHapley Additive exPlanations) for deeper explanations
   - Analyze feature importance using multiple methods
   - Create interactive dashboard to display explanations

4. **Deployment**:
   - Package model as API (FastAPI, Flask)
   - Deploy on cloud (AWS, Azure, GCP)
   - Build web interface for model usage

5. **Model Evaluation**:
   - Test model stability on new data
   - Set up monitoring to detect model drift
   - Regularly retrain model with new data

---

## 👥 Author & Contact

- **Project**: Capstone project for **PTDLKD** (Data Analysis and Mining)
- **University**: University of Information Technology (UIT - Year 3, Semester 1)

---

## 📚 References

- **LIME Paper**: [Why Should I Trust You?: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- **SHAP**: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **TensorFlow/Keras Documentation**: https://www.tensorflow.org/

---

## 📝 Notes

⚠️ **Important when running on Google Colab**:

- Must mount Google Drive before loading data
- Adjust file paths according to your directory structure
- Some cells may need adjustment depending on Colab configuration

✅ **Completed**: All 10+ models have been trained and stored in the `models/` directory

---

**Last Updated**: May, 2026
