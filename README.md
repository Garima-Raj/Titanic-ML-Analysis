# Titanic Survival Analysis 🚢 | Summer Training Project

This project is a part of my **Summer Training Program at Lovely Professional University (LPU)** under the guidance of **Ms. Gaurika Dhingra** as a proud member of the **Angaar Batch**.

## 🔍 Objective

To analyze the Titanic dataset using exploratory data analysis (EDA) and build machine learning models to predict passenger survival.


## 📁 Dataset

- Dataset used: `titanic.csv`  
- Contains information like Age, Gender, Passenger Class, Fare, and Survival status of Titanic passengers.


## 📊 Exploratory Data Analysis (EDA)

Key analysis steps:

- Checked for missing values and duplicates
- Encoded categorical columns (like Sex)
- Visualized survival patterns across features:
  - Survival distribution
  - Survival by Gender and Class
  - Age and Fare distributions
- Created a correlation heatmap for numerical insights
- Identified outliers using Z-score


## ⚙️ Machine Learning Models

Built and compared two classifiers:

1. **K-Nearest Neighbors (KNN)**
   - Tried multiple `k` values
   - Found the best k based on accuracy

2. **Decision Tree Classifier**
   - Used default scikit-learn implementation
   - Compared performance with KNN

### 📈 Results

- Decision Tree performed slightly better than KNN in this case.
- Gender, Age, and Class were key features influencing survival.


## 🛠️ Technologies Used

- Python
- Pandas
- Seaborn & Matplotlib
- Scikit-learn
- VS Code
- Git & GitHub


