# 🏦 Loan Approval Prediction using Decision Tree (ML Project)

This project is a simple supervised machine learning model that predicts whether a customer is likely to get a loan approved or not, based on historical loan application data.

---

## 📂 Files Included
- loan_approval_prediction_decision_tree.ipynb: Main Jupyter Notebook
- Dataset: [train.csv](#) (link not added here, please upload manually)
- Flowchart: Included inside notebook as visualization
- Live Prediction: Included to test with real-time user inputs

---

## 📌 Project Type
- ✅ Supervised Learning
- ✅ Classification Problem
- ✅ Decision Tree Algorithm

---

## ⚙️ Tools & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧠 Features Used
- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

---

## 📊 Model Used
- DecisionTreeClassifier() from sklearn

---

## ✅ Final Model Result

### ✔️ Accuracy:  
0.7560975609756098 (75.6%)

### ✔️ Confusion Matrix:
[[79  1] [29 14]]

### ✔️ Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Rejected) | 0.73 | 0.99 | 0.84 | 80 |
| 1 (Approved) | 0.93 | 0.33 | 0.48 | 43 |

- *Macro Avg*: Precision = 0.83, Recall = 0.66, F1-Score = 0.72  
- *Weighted Avg*: Precision = 0.80, Recall = 0.76, F1-Score = 0.76

---

## 🔍 Visualization

- ✅ Flowchart of the trained Decision Tree plotted using plot_tree() from sklearn.

---

## 🔮 Live Prediction

Tested with sample customer input:
```python
model.predict(new_df)

✅ Result:

Loan Status: Rejected
