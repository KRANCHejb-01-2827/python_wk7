import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

print("🔹 Loading dataset...")
# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
print("✅ Dataset loaded successfully!")

print("\n🔹 First 5 rows of the dataset:")
print(df.head())

print("\n🔹 Checking dataset info...")
print(df.info())

print("\n🔹 Checking for missing values:")
print(df.isnull().sum())

print("\n🔹 Basic statistics:")
print(df.describe())

print("\n🔹 Grouping by species and computing mean values...")
grouped = df.groupby("target").mean()
print(grouped)

# Visualization Section
print("\n📊 Creating visualizations...")

# Line chart
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.savefig("line_chart.png")
plt.show()
print("✅ Line chart saved as 'line_chart.png' and displayed")

# Bar chart
plt.figure(figsize=(8, 5))
grouped["sepal length (cm)"].plot(kind="bar", color="skyblue")
plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length (cm)")
plt.savefig("bar_chart.png")
plt.show()
print("✅ Bar chart saved as 'bar_chart.png' and displayed")

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["sepal length (cm)"], bins=15, color="orange", edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.show()
print("✅ Histogram saved as 'histogram.png' and displayed")

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.savefig("scatter_plot.png")
plt.show()
print("✅ Scatter plot saved as 'scatter_plot.png' and displayed")

print("\n🎉 Program finished! All charts saved & displayed.")
