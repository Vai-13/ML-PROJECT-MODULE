#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install xgboost')
#importing all the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


#For data preprocessing
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#For Model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#for evaluation
from sklearn.metrics import confusion_matrix,roc_curve,auc
import os


# In[6]:


# Replace 'Dentistry.csv' with the path to your dataset file
data = pd.read_csv('Dentistry.csv')
print(data.head())


# In[7]:


# Fill missing values with the mean of the respective columns
data.fillna(data.mean(), inplace=True)


# In[8]:


# Encode categorical data
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])


# In[9]:


X = data.drop('Gender', axis=1)  # Independent variables
Y = data['Gender']               # Dependent variable


# In[15]:


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Check for missing columns
original_columns = X.columns
if X_imputed.shape[1] != len(original_columns):
    missing_columns = set(original_columns) - set(X.columns)
    print(f"Missing columns: {missing_columns}")
    # Handle missing columns if necessary

# Normalize the imputed data
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X_imputed)

# Ensure the number of columns matches
X_normalized_df = pd.DataFrame(X_normalized, columns=original_columns[:X_imputed.shape[1]])

# Verify the columns
print(X_normalized_df.columns)


# In[18]:


correlation_matrix = X_normalized_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X_normalized_df, Y, test_size=0.2, random_state=42)


# In[20]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)


# In[21]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)


# In[22]:


random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)


# In[23]:


xgboost = XGBClassifier()
xgboost.fit(X_train, Y_train)


# In[25]:


from sklearn.metrics import accuracy_score

# Predicting and evaluating Logistic Regression model
Y_pred_lr = logistic_regression.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(Y_test, Y_pred_lr))


# In[26]:


Y_pred_dt = decision_tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(Y_test, Y_pred_dt))


# In[27]:


Y_pred_rf = random_forest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, Y_pred_rf))


# In[28]:


Y_pred_xgb = xgboost.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(Y_test, Y_pred_xgb))


# In[31]:


from sklearn.metrics import roc_auc_score, roc_curve

# Assuming xgboost and other variables are already defined

# Calculate the predicted probabilities and AUC score
Y_prob_xgb = xgboost.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(Y_test, Y_prob_xgb)
roc_auc_xgb = roc_auc_score(Y_test, Y_prob_xgb)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='blue', label=f'XGBoost AUC = {roc_auc_xgb:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




