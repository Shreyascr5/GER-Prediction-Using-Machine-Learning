# 🎓 GER Prediction Using Machine Learning

## 📌 Project Overview

This project develops a complete Machine Learning pipeline to predict the **Gross Enrollment Ratio (GER)** in higher education using educational indicators derived from AISHE (All India Survey on Higher Education) data.

The project follows an end-to-end ML lifecycle including:

- Data Understanding
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Feature Engineering
- Model Selection & Cross Validation
- Dimensionality Reduction (PCA)
- Clustering (K-Means)
- Hyperparameter Tuning
- Ensemble Learning
- Final Evaluation & Generalisation Analysis

The objective is to identify the key educational factors influencing GER and build an accurate predictive model for educational planning and policy analysis.

---

# 📂 Dataset

### Source

AISHE (All India Survey on Higher Education)

### Dataset Characteristics

| Feature              | Value |
| -------------------- | ----- |
| Rows                 | 143   |
| Columns              | 57    |
| Numerical Features   | 55    |
| Categorical Features | 2     |
| Duplicate Records    | 0     |

### Key Variables

- Enrollment Statistics
- Teacher Statistics
- Pupil Teacher Ratio
- Gender Parity Index (GPI)
- University Counts
- Social Category Enrollment
- State and Year Information

### Target Variable

```text
ger_all_total
```

Gross Enrollment Ratio (GER)

---

# 🛠️ Project Workflow

## Week 2 – Data Understanding

- Dataset inspection
- Data quality assessment
- Missing value analysis
- Duplicate record analysis
- Dataset profiling

---

## Week 3 – Exploratory Data Analysis

- Correlation analysis
- Distribution analysis
- Outlier detection
- Feature relationship exploration
- Summary statistics

---

## Week 4 – Statistical Analysis

- Skewness analysis
- Kurtosis analysis
- Normality testing
- Distribution classification
- Statistical interpretation

---

## Week 5 – Feature Engineering

### Engineered Features

- Student Teacher Ratio
- Teachers per University
- Students per University
- Female Participation Ratio
- PG Ratio
- PhD Ratio

### Transformations

- Log Transformations
- Feature Importance Analysis
- Correlation Comparison

---

## Week 6 – Model Comparison & Cross Validation

### Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Validation Techniques

- Train-Test Split Comparison
  - 70-30
  - 75-25
  - 80-20
  - 85-15

- Feature Scaling Comparison
  - StandardScaler
  - MinMaxScaler

- K-Fold Cross Validation
  - K = 2 to 7

### Best Results

| Model             | R²    |
| ----------------- | ----- |
| Random Forest     | 0.895 |
| Gradient Boosting | 0.817 |
| Decision Tree     | 0.724 |

---

## Week 7 – PCA & K-Means Clustering

### PCA

- Standardization
- Principal Component Analysis
- Explained Variance Analysis
- Cumulative Variance Analysis

### Results

- Original Features: 95
- PCA Components: 35
- Variance Retained: 95%

### K-Means Clustering

- K = 2 to 7 comparison
- Elbow Method
- Silhouette Analysis

### Best Cluster Count

```text
K = 2
```

---

## Week 8 – Hyperparameter Tuning & Ensemble Learning

### Hyperparameter Tuning

#### Grid Search

- Random Forest
- Gradient Boosting

#### Randomized Search

- Random Forest
- Gradient Boosting

### Ensemble Methods

| Model             | Ensemble Type |
| ----------------- | ------------- |
| Decision Tree     | Base Model    |
| Random Forest     | Bagging       |
| Gradient Boosting | Boosting      |
| Voting Regressor  | Voting        |

### Final Ensemble Ranking

| Rank | Model             | R²    |
| ---- | ----------------- | ----- |
| 1    | Random Forest     | 0.890 |
| 2    | Voting Regressor  | 0.886 |
| 3    | Gradient Boosting | 0.882 |
| 4    | Decision Tree     | 0.724 |

---

## Week 9 – Final Evaluation & Launch

### Final Model

```text
Random Forest Regressor
```

### Final Performance

| Metric   | Value  |
| -------- | ------ |
| Train R² | 0.9684 |
| Test R²  | 0.8901 |
| MAE      | 2.83   |
| RMSE     | 3.87   |
| MAPE     | 10.32% |

### Generalisation Analysis

| Metric               | Value  |
| -------------------- | ------ |
| Generalisation Error | 0.0783 |

Interpretation:

- Good Generalisation
- Limited Overfitting
- Strong Predictive Capability

### Additional Evaluation

- Learning Curve Analysis
- Residual Analysis
- Actual vs Predicted Comparison
- Confusion Matrix (Categorized GER Levels)

---

# 🏆 Best Model

## Random Forest Regressor

### Performance

```text
Train R²  = 0.9684
Test R²   = 0.8901
RMSE      = 3.8675
MAE       = 2.8296
MAPE      = 10.32%
```

### Why It Was Selected

- Highest Test R²
- Lowest RMSE
- Strong Generalisation
- Robust Against Overfitting
- Best Overall Predictive Performance

---

# 📊 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

### Machine Learning Techniques

- Regression Modeling
- Feature Engineering
- Cross Validation
- PCA
- K-Means Clustering
- Hyperparameter Tuning
- Ensemble Learning

### Development Tools

- Git
- GitHub
- VS Code

---

# 📈 Key Achievements

✅ Complete End-to-End Machine Learning Pipeline

✅ Feature Engineering and Statistical Analysis

✅ PCA Dimensionality Reduction

✅ K-Means Clustering with Elbow & Silhouette Analysis

✅ Hyperparameter Tuning using Grid Search and Randomized Search

✅ Ensemble Learning (Voting, Bagging, Boosting)

✅ Strong Final Performance (R² = 0.8901)

✅ Generalisation Error Analysis

✅ Industry-Style Project Structure

---

# 👥 Project Team

**Team 2 – M.Tech Data Science**

**Department of Artificial Intelligence & Data Science**
**M.S. Ramaiah Institute of Technology (MSRIT), Bengaluru**

### Team Details

This project was developed as part of the Machine Learning coursework under the M.Tech Data Science program at MSRIT.

### Project Title

**GER Prediction Using Machine Learning**

### Academic Program

**M.Tech Data Science**
**M.S. Ramaiah Institute of Technology (MSRIT)**

# 📜 License

This project is developed for educational and academic purposes.
