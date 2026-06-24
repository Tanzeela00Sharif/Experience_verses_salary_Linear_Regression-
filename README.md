# Experience_verses_salary_Linear_Regression-
# Salary Prediction using Linear Regression

A simple machine learning project that predicts an employee's salary based on their years of experience using **Simple Linear Regression**.

## 📌 Project Overview

This project demonstrates a basic supervised machine learning workflow:
- Loading and exploring a dataset
- Splitting data into training and testing sets
- Training a Linear Regression model
- Evaluating model performance
- Making predictions on new data

## 📊 Dataset

The dataset contains two columns:

| Column | Description |
|---|---|
| `YearsExperience` | Number of years of professional experience |
| `Salary` | Annual salary (in USD) |

**Source:** [Salary Data - Simple Linear Regression (Kaggle)](https://www.kaggle.com/datasets)

## 🛠️ Tech Stack

- Python 3
- pandas
- scikit-learn
- matplotlib (for visualization)
- Google Colab

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/salary-prediction-linear-regression.git
cd salary-prediction-linear-regression
pip install -r requirements.txt
```

**requirements.txt**
## 🚀 Usage

1. Load the dataset:
```python
import pandas as pd
df = pd.read_csv('Salary_Data.csv')
```

2. Split features and target:
```python
X = df[['YearsExperience']]
y = df['Salary']
```

3. Split into train/test sets:
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Train the model:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
```

5. Make a prediction:
```python
model.predict([[11.2]])  # Predicts salary for 11.2 years of experience
```

## 📈 Model Evaluation

The model is evaluated using:
- **R² Score** — measures how well the model explains the variance in salary
- **Mean Squared Error (MSE)** — measures average prediction error

```python
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(x_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

## 📉 Visualization

A scatter plot with the regression line is used to visualize the relationship between years of experience and salary.

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(x_test, model.predict(x_test), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.legend()
plt.show()
```

## 📂 Project Structure
