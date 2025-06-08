import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Load Iris dataset from seaborn-data (CSV format)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print(df.head())

# 1. Histogram
df.hist(figsize=(10, 6), edgecolor='black')
plt.suptitle('Histograms of All Features', fontsize=16)
plt.tight_layout()
plt.show()

# 2. Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :-1], palette="Set2")
plt.title("Boxplot of Iris Features")
plt.show()

# 3. Violin Plot
plt.figure(figsize=(10, 6))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x="species", y=column, data=df)
    plt.title(f'Violin Plot - {column}')
plt.tight_layout()
plt.show()

# 4. Scatter Plot
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.show()

# 5. Pair Plot
sns.pairplot(df, hue="species", palette="Set1")
plt.suptitle("Pair Plot of All Features", y=1.02)
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 7. Bar Plot (count of species)
sns.countplot(x="species", data=df, palette="muted")
plt.title("Count of Each Iris Species")
plt.show()

# 8. Pie Chart
species_counts = df['species'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Pie Chart of Iris Species")
plt.show()

# 9. 3D Scatter Plot
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                    color='species', title="3D Scatter Plot of Iris Features")
fig.show()

# 10. Line Plot (feature trend by index)
plt.figure(figsize=(10, 6))
for column in df.columns[:-1]:
    plt.plot(df[column], label=column)
plt.legend()
plt.title("Line Plot of All Features Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.show()

# 11. Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x="species", y="petal_length", data=df, palette="Set2")
plt.title("Swarm Plot - Petal Length by Species")
plt.show()
