# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# No missing values in Iris dataset, but we demonstrate how to drop/clean if needed
# df.dropna(inplace=True)  # Uncomment if using a dataset with missing values

# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping: Mean of numerical columns per species
grouped = df.groupby('species').mean()
print("\nMean of features by species:")
print(grouped)

# ------------------------------
# Task 3: Data Visualizations
# ------------------------------

# Set style
sns.set(style="whitegrid")

# 1. Line chart: Average feature values per species
grouped.T.plot(kind='line', marker='o')
plt.title("Average Feature Values per Iris Species")
plt.xlabel("Features")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# 2. Bar chart: Average petal length per species
sns.barplot(data=df, x='species', y='petal length (cm)', ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram: Sepal length distribution
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: Sepal Length vs Petal Length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

# Observations
print("\nObservations:")
print("- Setosa species tend to have smaller petal lengths and widths.")
print("- Versicolor and Virginica show more overlap but can be separated using petal dimensions.")
print("- Distribution of sepal length is approximately normal.")
print("- Clear positive correlation between sepal and petal length.")

