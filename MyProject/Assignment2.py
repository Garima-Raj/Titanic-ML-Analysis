import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import zscore

df = pd.read_csv(r"C:\Users\Garima Raj\Downloads\titanic.csv")

print("Shape of dataset:", df.shape)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Dropping the 'Name' column
df.drop(columns=["Name"], inplace=True)

# Removing rows with missing Age values
df.dropna(subset=["Age"], inplace=True)

# Encoding the 'Sex' column
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])

# Outlier detection using Z-score
for col in ["Age", "Fare"]:
    z = np.abs(zscore(df[col]))
    print(f"Outliers in '{col}':", (z > 3).sum())

# Countplot: Survival distribution
sns.countplot(x="Survived", data=df)
plt.title("Survival Distribution")
plt.show()

# Survival by Gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xticks([0, 1], ["Female", "Male"])
plt.show()

# Survival by Passenger Class
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution with survival
sns.histplot(data=df, x="Age", hue="Survived", kde=True)
plt.title("Age Distribution by Survival")
plt.show()

# Fare vs Survival using Boxplot
sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare by Survival")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluating KNN for multiple values of k
knn_scores = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    knn_scores.append(acc)
    print(f"K = {k}, Accuracy = {acc:.4f}")

# Finding the best K
best_k = knn_scores.index(max(knn_scores)) + 1

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

# Final model comparison
print("\nBest K for KNN:", best_k)
print(f"KNN Accuracy: {knn_scores[best_k-1]:.4f}")
print(f"Decision Tree Accuracy: {dt_acc:.4f}")

if dt_acc > knn_scores[best_k-1]:
    print("Decision Tree performed better on this dataset.")
elif dt_acc < knn_scores[best_k-1]:
    print("KNN performed better on this dataset.")
else:
    print("Both models performed equally well.")

# Classification reports
print("\nKNN Classification Report:")
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
print(classification_report(y_test, best_knn.predict(X_test)))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_preds))
