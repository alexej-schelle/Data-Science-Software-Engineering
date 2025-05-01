#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import datasets
iris = datasets.load_iris()
input_data = iris.data
output_data = iris.target


# In[4]:


X_sepal_length = input_data[:,0]
X_sepal_width = input_data[:,1]
X_petal_length = input_data[:,2]
X_sepal_width = input_data[:,3]


# In[7]:


from sklearn.model_selection import train_test_split
input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size = 0.4)


# In[9]:


input_data_train.shape, output_data_train.shape, input_data.shape, output_data.shape


# In[11]:


from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(1)


# In[13]:


clf.fit(input_data_train, output_data_train)


# In[15]:


clf.predict([[6.3, 2.7, 5.5, 1.5]])


# In[17]:


clf.score(input_data_train, output_data_train)


# In[21]:


clf.score(input_data_test, output_data_test)


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the iris dataset directly from seaborn
df = sns.load_dataset("iris")

sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()


# In[26]:


features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'{feature.capitalize()} by Species')
plt.tight_layout()
plt.show()


# In[28]:


from pandas.plotting import parallel_coordinates

plt.figure(figsize=(10, 6))
parallel_coordinates(df, class_column="species", cols=features, color=["#556b2f", "#8fbc8f", "#4682b4"])
plt.title("Parallel Coordinates Plot")
plt.show()


# In[30]:


import numpy as np

# Calculate mean per species
mean_df = df.groupby('species')[features].mean()

# Radar chart function
def radar_chart(ax, data, feature_labels, species):
    angles = np.linspace(0, 2 * np.pi, len(feature_labels), endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]
    
    ax.plot(angles, data, color='tab:blue', linewidth=2, label=species)
    ax.fill(angles, data, color='tab:blue', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels)

# Plot radar charts for each species
fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(15, 5))
for i, (species, row) in enumerate(mean_df.iterrows()):
    radar_chart(axes[i], row.tolist(), features, species)
    axes[i].set_title(species.capitalize(), size=14)
plt.suptitle("Radar Chart - Feature Averages per Species")
plt.show()


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load iris dataset
df = sns.load_dataset("iris")

# Use only sepal features
X = df[['sepal_length', 'sepal_width']].values
y = LabelEncoder().fit_transform(df['species'])  # Encode species as integers

# Train SVM classifier
clf = SVC(kernel='linear', random_state=0)
clf.fit(input_data, output_data)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict for the grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', edgecolor='k', s=80)

# Legend
legend_labels = df['species'].unique()
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(i), markersize=10) for i in range(3)]
plt.legend(handles, legend_labels, title='Species')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Decision Regions for Sepal Features (SVM)')
plt.show()


# In[42]:


import pandas as pd

# Load iris dataset
df = sns.load_dataset("iris")


# In[64]:


print(df)


# In[44]:


df.head()


# In[46]:


df.count()


# In[48]:


df['sepal_width']


# In[50]:


df['sepal_width'].isnull()


# In[52]:


df.dropna()


# In[54]:


print(df)


# In[56]:


iris_versicolor = df[df['species'] == 'setosa']


# In[58]:


print(iris_versicolor['sepal_length'].mean())


# In[60]:


df.loc[82]


# In[62]:


df.iloc[82]


# In[64]:


df.count()


# In[66]:


df.groupby('species').count()


# In[68]:


df.groupby('species').describe()


# In[70]:


df.groupby('species').hist()


# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("iris")

# Correct usage
sns.jointplot(data=df, x='sepal_length', y='petal_length', kind='scatter')

plt.show()


# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("iris")

# Compute correlation matrix using only numerical columns (excluding 'species')
corrmat = df.drop('species', axis=1).corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corrmat, annot=True, cmap='coolwarm', fmt='.2f', square=True)

plt.title("Feature Correlation Heatmap")
plt.show()


# In[76]:


from sklearn.datasets import load_digits

data = load_digits()
input_data, output_data = data.data, data.target

input_data.shape
output_data.shape


# In[78]:


import sklearn.linear_model as lm
lr = lm.LinearRegression()
lr.fit(input_data, output_data)
lr.score(input_data, output_data)


# In[80]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[82]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[84]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[86]:


import sklearn.linear_model as lm
logr = lm.LogisticRegression()

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

input_data = diabetes.data
output_data = diabetes.target

input_data.shape
output_data.shape


# In[88]:


logr.fit(input_data, output_data)


# In[90]:


logr.score(input_data, output_data)


# In[98]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = load_diabetes()

input_data = diabetes.data
output_data = diabetes.target

model1 = SVC(kernel='linear', C=1)
model1.fit(input_data, output_data)
model1.score(input_data, output_data)


# In[302]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[104]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



model = SVC(kernel='rbf', C=1E6, gamma=1.0)
model.fit(X, y)
print(model.score(X,y))


# In[106]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[108]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[110]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[112]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[138]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Print actual vs predicted values
print("Actual vs Predicted values:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# In[140]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X,y)
print(tree.score(X,y))


# In[142]:


from sklearn.tree import DecisionTreeClassifier
import matplotlib

tree = DecisionTreeClassifier(max_depth = 1)
tree.fit(X,y)
print(tree.score(X,y))

tree2 = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 1, max_leaf_nodes = 8)
tree2.fit(X,y)
print(tree2.score(X,y))

tree3 = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 1, min_samples_split = 4, max_leaf_nodes = 8)
tree3.fit(X,y)
print(tree3.score(X,y))


# In[144]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

RandomForestClassifier(bootstrap=True, class_weight=None, criterion = 'gini', max_depth = None, max_features = 'auto', max_leaf_nodes = None, min_samples_leaf = 1, min_samples_split = 2, min_weight_fraction_leaf = 0.0, n_estimators = 10, n_jobs = 1, oob_score=False, random_state = None, verbose = 0, warm_start= False)

rf.fit(X, y)
print(rf.score(X,y))


# In[146]:


# Boosted Decision Trees

from sklearn.ensemble import AdaBoostClassifier
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

# Explicitly set algorithm to 'SAMME' to avoid warning
clf = AdaBoostClassifier(algorithm="SAMME")

bdt = AdaBoostClassifier()
bdt.fit(X,y)

print(bdt.score(X,y))


# In[148]:


import numpy as np
import matplotlib.pyplot as plt

n = 100; vmin = 0; vmax = 10
x1 = np.random.uniform(vmin, vmax, n)
x2 = np.random.uniform(vmin, vmax, n)
x3 = np.random.uniform(vmin, vmax, n)

# Plot histogram
plt.hist(x1, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.hist(x2, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.hist(x3, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.scatter(x1, x2, color='blue', marker='o')

# Labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Basic Scatter Plot')

# Show plot
plt.show()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x1, x2, x3, c='green', marker='o')

# Labels
ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('X3 Label')
ax.set_title('3D Scatter Plot')

plt.show()


# In[150]:


from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC

svc = SVC(kernel='linear')
iris = load_iris()

for fid in range(len(iris.feature_names)):
    
    X = iris.data[:,fid,np.newaxis]
    y = iris.target

    y[y==2] = 1
    clf = svc.fit(X,y)
    print(iris.feature_names[fid], 'univariate score: '), clf.score(X,y)


# In[190]:


# Decision and selection based on Correlations and Accuracy

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the iris dataset from sklearn
iris = datasets.load_iris()

y = iris.target
X = iris.data

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

print(df.describe()) # Statistical Measures
print(df.corr()) # Normalized Correlations
print(df.corr('pearson')) # Pearson Correlations
print(df.corr('spearman')) # Spearman Correlations
print(df.cov()) # Covariance Matrix


# In[154]:


# Calculating Scores for Linear Regression

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


# In[156]:


# Calculating accuracy score for classification

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets

# Load iris dataset
iris = datasets.load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # 0 = setosa, 1 = versicolor, 2 = virginica

# Features and target
X = df[iris.feature_names]
y = df['species']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[164]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load iris dataset
diabetes = datasets.load_diabetes()

# Create DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['species'] = diabetes.target  # 0 = setosa, 1 = versicolor, 2 = virginica

# Features and target
X = df[diabetes.feature_names]
y = df['species']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[166]:


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


# In[172]:


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


# In[188]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import classification_report, precision_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter("ignore", category=UndefinedMetricWarning)

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

