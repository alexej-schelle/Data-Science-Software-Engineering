#!/usr/bin/env python
# coding: utf-8

# In[43]:


import seaborn as sns
diamonds = sns.load_dataset('diamonds')
print(diamonds.head())


# In[53]:


var_carat = diamonds['carat']
var_cut = diamonds['cut']
var_color = diamonds['color']
var_clarity = diamonds['clarity']
var_depth = diamonds['depth']
var_table = diamonds['table']
var_price = diamonds['price']
var_x = diamonds['x']
var_y = diamonds['y']
var_z = diamonds['z']


# In[49]:


print(var_carat)


# In[55]:


print(var_cut)


# In[57]:


print(var_color)


# In[59]:


print(var_clarity)


# In[61]:


print(var_color)


# In[63]:


print(var_clarity)


# In[65]:


print(var_depth)


# In[67]:


print(var_table)


# In[69]:


print(var_price)


# In[71]:


print(var_x)


# In[73]:


print(var_y)


# In[75]:


print(var_z)


# In[134]:


import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
diamonds = sns.load_dataset('diamonds')

# Encode categorical columns (cut, color, clarity) using LabelEncoder
label_encoders = {}
categorical_cols = ['cut', 'color', 'clarity']

for col in categorical_cols:
    le = LabelEncoder()
    diamonds[col] = le.fit_transform(diamonds[col])
    label_encoders[col] = le  # Save encoder for later (optional)

# Features (X) and Target (y)
X = diamonds[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = diamonds['price']

# Train-test split
input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model and fit
model = LinearRegression()
model.fit(input_data_train, output_data_train)

# Predictions
output_data_pred = model.predict(input_data_test)

# Evaluate the model
mse = mean_squared_error(output_data_test, output_data_pred)
r2 = r2_score(output_data_test, output_data_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[144]:


# Make prediciton

new_diamond = pd.DataFrame([{
    'carat': 0.5,
    'cut': label_encoders['cut'].transform(['Ideal'])[0],
    'color': label_encoders['color'].transform(['G'])[0],
    'clarity': label_encoders['clarity'].transform(['SI1'])[0],
    'depth': 61.5,
    'table': 55.0,
    'x': 4.0,
    'y': 4.1,
    'z': 2.5
}])

print(new_diamond)

predicted_price = model.predict(new_diamond)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")


# In[9]:


import seaborn as sns
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
diamonds = sns.load_dataset('diamonds')

# Encode categorical columns (each column gets its own encoder)
le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()

diamonds['cut'] = le_cut.fit_transform(diamonds['cut'])
diamonds['color'] = le_color.fit_transform(diamonds['color'])
diamonds['clarity'] = le_clarity.fit_transform(diamonds['clarity'])

# Features and target
X = diamonds[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = diamonds['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (helps with KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Classifier
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Example prediction - new diamond
new_diamond = pd.DataFrame([{
    'carat': 0.3,
    'cut': le_cut.transform(['Ideal'])[0],   # use transform, NOT fit_transform
    'color': le_color.transform(['J'])[0],
    'clarity': le_clarity.transform(['SI2'])[0],
    'depth': 61.0,
    'table': 55.1,
    'x': 1.2,
    'y': 1.3,
    'z': 1.4
}])

# Scale the new diamond data
new_diamond_scaled = scaler.transform(new_diamond)

# Predict the price category
prediction = clf.predict(new_diamond_scaled)
print("Predicted price:", prediction[0], 'Dollar')

