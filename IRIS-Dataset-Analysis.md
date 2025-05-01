# Load dataset from database

from sklearn import datasets
iris = datasets.load_iris()
input_data = iris.data
output_data = iris.target

# Subdivide into different components of the dataset

X_sepal_length = input_data[:,0]
X_sepal_width = input_data[:,1]
X_petal_length = input_data[:,2]
X_sepal_width = input_data[:,3]

# Split the dataset

from sklearn.model_selection import train_test_split
input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size = 0.4)

# Check the data shape

input_data_train.shape, output_data_train.shape, input_data.shape, output_data.shape

# Nearest Neighbour Classification

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(1)
clf.fit(input_data_train, output_data_train)

clf.predict([[6.3, 2.7, 5.5, 1.5]])

# Training cycle

clf.score(input_data_train, output_data_train) 

# Validation cycle

clf.score(input_data_test, output_data_test) 

# Load and show the iris dataset directly from seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset("iris")

sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Show data ranges

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'{feature.capitalize()} by Species')
plt.tight_layout()
plt.show()

# Highlight parameter space

import numpy as np

mean_df = df.groupby('species')[features].mean()

def radar_chart(ax, data, feature_labels, species):
    angles = np.linspace(0, 2 * np.pi, len(feature_labels), endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]
    
    ax.plot(angles, data, color='tab:blue', linewidth=2, label=species)
    ax.fill(angles, data, color='tab:blue', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels)

fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(15, 5))
for i, (species, row) in enumerate(mean_df.iterrows()):
    radar_chart(axes[i], row.tolist(), features, species)
    axes[i].set_title(species.capitalize(), size=14)
plt.suptitle("Radar Chart - Feature Averages per Species")
plt.show()

# Classification with SVC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset("iris")

input_data = df[['sepal_length', 'sepal_width']].values
output_data = LabelEncoder().fit_transform(df['species'])  # Encode species as integers

clf = SVC(kernel='linear', random_state=0)
clf.fit(input_data, output_data)

a_min, a_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
b_min, b_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', edgecolor='k', s=80)

legend_labels = df['species'].unique()
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(i), markersize=10) for i in range(3)]
plt.legend(handles, legend_labels, title='Species')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Decision Regions for Sepal Features (SVM)')
plt.show()

# Analytics in pandas framework

import pandas as pd

df = sns.load_dataset("iris")
print(df)
df.head()
df.count()
df['sepal_width']
df['sepal_width'].isnull()
df.dropna()
print(df)

iris_versicolor = df[df['species'] == 'setosa']
print(iris_versicolor['sepal_length'].mean())

df.loc[82]
df.iloc[82]
df.count()

df.groupby('species').count()
df.groupby('species').describe()

df.groupby('species').hist()

# Scatter plot in Seaborn Library

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

sns.jointplot(data=df, x='sepal_length', y='petal_length', kind='scatter')

plt.show()

# Feature correlation map

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

corrmat = df.drop('species', axis=1).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corrmat, annot=True, cmap='coolwarm', fmt='.2f', square=True)

plt.title("Feature Correlation Heatmap")
plt.show()

# The same works for other types of correlations

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', square=True)
plt.title('Confusion Matrix - Iris Dataset')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Linear Regression Model

from sklearn.datasets import load_digits

data = load_digits()
input_data, output_data = data.data, data.target

input_data.shape
output_data.shape

import sklearn.linear_model as lm
lr = lm.LinearRegression()
lr.fit(input_data, output_data)
lr.score(input_data, output_data)

# Show linear regression in a graphical figure

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()

# Decision Tree classifier

from sklearn.tree import DecisionTreeClassifier
import matplotlib

tree = DecisionTreeClassifier()
tree.fit(X,y)
print(tree.score(X,y))

tree = DecisionTreeClassifier(max_depth = 1)
tree.fit(X,y)
print(tree.score(X,y))

tree2 = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 1, max_leaf_nodes = 8)
tree2.fit(X,y)
print(tree2.score(X,y))

tree3 = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 1, min_samples_split = 4, max_leaf_nodes = 8)
tree3.fit(X,y)
print(tree3.score(X,y))

# Vorhersagen mit Decision Tree Classifier für den IRIS Datensatz
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Iris-Datensatz laden
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Trainings- und Testsplit (optional, hier nutzen wir alle Daten)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell 1: Einfacher Baum mit max_depth=1
tree1 = DecisionTreeClassifier(max_depth=1)
tree1.fit(X_train, y_train)

# Vorhersagen und Genauigkeit
y_pred1 = tree1.predict(X_test)
print("\nTree 1 - Depth 1")
print("Score (Train):", tree1.score(X_train, y_train))
print("Score (Test):", tree1.score(X_test, y_test))
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

# Modell 2: Tieferer Baum mit Einschränkungen
tree2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, max_leaf_nodes=8)
tree2.fit(X_train, y_train)

y_pred2 = tree2.predict(X_test)
print("\nTree 2 - Depth 5, max_leaf_nodes=8")
print("Score (Train):", tree2.score(X_train, y_train))
print("Score (Test):", tree2.score(X_test, y_test))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

# Modell 3: Noch mehr Constraints
tree3 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=4, max_leaf_nodes=8)
tree3.fit(X_train, y_train)

y_pred3 = tree3.predict(X_test)
print("\nTree 3 - Depth 5, max_leaf_nodes=8, min_samples_split=4")
print("Score (Train):", tree3.score(X_train, y_train))
print("Score (Test):", tree3.score(X_test, y_test))
print(classification_report(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))

# Decision and Selection based on Correlations and Accuracy

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

iris = datasets.load_iris()

y = iris.target
X = iris.data

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

print(df.describe()) # Statistical Measures
print(df.corr()) # Normalized Correlations
print(df.corr('pearson')) # Pearson Correlations
print(df.corr('spearman')) # Spearman Correlations
print(df.cov()) # Covariance Matrix

y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(y_train, X_train)

X_pred = model.predict(y_test)

# Example: Calculate score for linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Quick look at the data
print(df.head())

# Feature and target
X = df[['sepal length (cm)']]  # Independent variable
y = df['petal length (cm)']    # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_[0]}')

# Performance metrics
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R² Score: {r2_score(y_test, y_pred):.2f}')

plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

# Calculating accuracy score for classification

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load diabetes dataset
diabetes = datasets.load_diabetes()

# Create DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['species'] = diabetes.target  # target: 0, 1, 2 (setosa, versicolor, virginica)

# Features and target
X = df[diabetes.feature_names]
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVM model (classifier)
svm_model = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy:.2f}')

# Compare accuracy score to standard score (the same for classification models)
print(svm_model.score(X_test, y_test))

# PCA of IRIS Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# IRIS-Datensatz laden
iris = datasets.load_iris()
X = iris.data  # 4 numerische Merkmale
y = iris.target  # Zielklasse (0, 1, 2)

# PCA auf 2 Dimensionen reduzieren
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Ergebnisse in DataFrame für einfachere Handhabung
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Farben und Klassenbezeichnungen
colors = ['red', 'green', 'blue']
labels = ['Setosa', 'Versicolor', 'Virginica']

# Plot
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    subset = df_pca[df_pca['target'] == i]
    plt.scatter(subset['PC1'], subset['PC2'], color=colors[i], label=label)

plt.title('PCA des IRIS-Datensatzes')
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.legend()
plt.grid(True)
plt.show()

# Compare graphically to standard variance
iris = datasets.load_iris()
X = iris.data
y = iris.target

# In DataFrame umwandeln für einfache Visualisierung
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Mapping von Zahlen auf Klassennamen
target_names = dict(enumerate(iris.target_names))
df['species'] = df['target'].map(target_names)

# Scatterplot Matrix (Paarplots)
sns.pairplot(df, hue='species', diag_kind='kde', palette='Set1')
plt.show()

# Optionale Information: Varianz-Anteil pro Komponente
explained_variance = pca.explained_variance_ratio_
print(f'Erklärte Varianz: {explained_variance}')
print(f'Kumulative erklärte Varianz: {np.cumsum(explained_variance)}')

# Make predictions with DecisionTreeClassifier()

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Iris-Datensatz laden
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Trainings- und Testsplit (optional, hier nutzen wir alle Daten)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell 1: Einfacher Baum mit max_depth=1
tree1 = DecisionTreeClassifier(max_depth=1)
tree1.fit(X_train, y_train)

# Vorhersagen und Genauigkeit
y_pred1 = tree1.predict(X_test)
print("\nTree 1 - Depth 1")
print("Score (Train):", tree1.score(X_train, y_train))
print("Score (Test):", tree1.score(X_test, y_test))
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

# Modell 2: Tieferer Baum mit Einschränkungen
tree2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, max_leaf_nodes=8)
tree2.fit(X_train, y_train)

y_pred2 = tree2.predict(X_test)
print("\nTree 2 - Depth 5, max_leaf_nodes=8")
print("Score (Train):", tree2.score(X_train, y_train))
print("Score (Test):", tree2.score(X_test, y_test))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

# Modell 3: Noch mehr Constraints
tree3 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=4, max_leaf_nodes=8)
tree3.fit(X_train, y_train)

y_pred3 = tree3.predict(X_test)
print("\nTree 3 - Depth 5, max_leaf_nodes=8, min_samples_split=4")
print("Score (Train):", tree3.score(X_train, y_train))
print("Score (Test):", tree3.score(X_test, y_test))
print(classification_report(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
