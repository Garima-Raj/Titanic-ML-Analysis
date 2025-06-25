import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\Garima Raj\Downloads\Iris.csv")

print("Info: ",df.info)
print("Describe: ",df.describe())
df.drop(columns=["Id"], inplace=True)
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Outlier detection using Z-score
z_scores = np.abs(stats.zscore(df.select_dtypes(include='number')))
outliers = (z_scores > 3).sum(axis=0)
print("Outliers per column:\n", outliers)

# Countplot for Species
sns.countplot(data=df, x="Species")
plt.title("Species Distribution")
plt.show()

# Boxplots
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
plt.figure(figsize=(10, 6))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


# Scatterplots
sns.scatterplot(data=df, x="SepalLengthCm", y="PetalLengthCm", hue="Species", palette="Set2")
plt.title("Sepal Length vs Petal Length")
plt.show()

sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", palette="Set2")
plt.title("Sepal Length vs Sepal Width")
plt.show()

# Pairplot
sns.pairplot(df, hue="Species", palette="Set2", diag_kind="kde")
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

# Label encode Species
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Split data
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Trying different values of k
accuracy_scores = []
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    print(f"K = {k}, Accuracy = {acc:.4f}")

# Plotting k vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), accuracy_scores, marker='o', linestyle='--', color='blue')
plt.title("KNN Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks(range(1, 16))
plt.grid(True)
plt.show()

# Final model with best k
best_k = accuracy_scores.index(max(accuracy_scores)) + 1
print(f"\nBest K = {best_k}")

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
final_pred = final_model.predict(X_test_scaled)

print("\nClassification Report:\n", classification_report(y_test, final_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_pred))
