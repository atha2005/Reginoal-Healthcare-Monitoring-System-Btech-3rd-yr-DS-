#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 1. Data Loading

# In[6]:


file_path = r"E:\3yr Project\dataset\archive\updated_lung_dataset.csv"
df = pd.read_csv(file_path)


# # 2. Data Inspection & Cleaning

# In[7]:


print("First 5 rows:\n", df.head())


# In[8]:


dtype_counts = df.dtypes.value_counts()
labels = dtype_counts.index.astype(str)  # Convert dtype index to strings

plt.figure(figsize=(6,6))
plt.pie(dtype_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Data Types Distribution')
plt.show()
print(df.info())


# In[9]:


print("\nData types:\n", df.dtypes)


# In[10]:


print("\nNull values:\n", df.isnull().sum())


# # 3. Aggregation & Descriptive Stats

# In[11]:


df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


# # a) State-wise total/average hospital visits

# In[12]:


grouped_df = df.groupby('State')['Hospital Visits'].agg(['sum', 'mean']).reset_index()

plt.figure(figsize=(12,6))
plt.bar(grouped_df['State'], grouped_df['sum'], color='skyblue')
plt.xlabel('State')
plt.ylabel('Total Hospital Visits')
plt.title('State-wise Total Hospital Visits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Similarly, for average:
plt.figure(figsize=(12,6))
plt.bar(grouped_df['State'], grouped_df['mean'], color='orange')
plt.xlabel('State')
plt.ylabel('Average Hospital Visits')
plt.title('State-wise Average Hospital Visits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\nState-wise sum/avg hospital visits:")
print(df.groupby('State')['Hospital Visits'].agg(['sum', 'mean']).reset_index())


# # b) State and Disease Type-wise hospital visits

# In[13]:


grouped_data = df.groupby(['State', 'Disease Type'])['Hospital Visits'].sum().reset_index()

pivot_table = grouped_data.pivot(index='State', columns='Disease Type', values='Hospital Visits')

# Plot grouped bar chart
pivot_table.plot(kind='bar', figsize=(12, 6))

plt.title('Total Hospital Visits by State and Disease Type')
plt.xlabel('State')
plt.ylabel('Total Hospital Visits')
plt.xticks(rotation=45)
plt.legend(title='Disease Type')
plt.tight_layout()
plt.show()
print("\nVisits by State & Disease Type:")
print(df.groupby(['State', 'Disease Type'])['Hospital Visits'].sum().reset_index())


# # c) Calculate and print recovery rate per state for a disease (e.g., COPD)

# In[14]:


disease_df = df[df['Disease Type'] == 'COPD'].copy()
disease_df['Recovered'] = disease_df['Recovered'].map({'Yes': 1, 'No': 0})
state_recovery = disease_df.groupby('State')['Recovered'].agg(['sum', 'count']).reset_index()
state_recovery['Recovery Rate (%)'] = (state_recovery['sum'] / state_recovery['count']) * 100

print("\nTop 5 states by COPD recovery rate:")
print(state_recovery[['State', 'Recovery Rate (%)']].sort_values('Recovery Rate (%)', ascending=False).head(5))
top_states = state_recovery[['State', 'Recovery Rate (%)']].sort_values('Recovery Rate (%)', ascending=False).head(5)

print("\nTop 5 states by COPD recovery rate:")
print(top_states)

plt.figure(figsize=(10,6))
plt.bar(top_states['State'], top_states['Recovery Rate (%)'], color='skyblue')
plt.title('Top 5 States by COPD Recovery Rate')
plt.xlabel('State')
plt.ylabel('Recovery Rate (%)')
plt.ylim(0, 100)

# To show values above bars
for index, value in enumerate(top_states['Recovery Rate (%)']):
    plt.text(index, value + 1, f"{value:.2f}%", ha='center')

plt.show()


# # 4. Data Prep for ML

# # Encode categorical columns

# In[15]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[16]:


label_cols = ['State', 'Disease Type', 'Treatment Type', 'Chronic_Infection', 'Smoking Status', 'Gender']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))


# # Define features and target

# In[17]:


features = [
    'Age', 'Gender', 'Smoking Status', 'Lung Capacity',
    'Oxygen_Level', 'Disease Type', 'Treatment Type',
    'Hospital Visits', 'Chronic_Infection', 'State'
]
target = 'Recovered'
df[target] = df['Recovered'].map({'Yes': 1, 'No': 0}) if df['Recovered'].dtype == 'O' else df['Recovered']


# In[18]:


X = df[features]
y = df[target]


# # Feature scaling (important for ML)

# In[19]:


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# # 5. Train/Test Split

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=80)


# # 6. Train and Evaluate ML Models

# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[22]:


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=45),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}


# In[23]:


results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    # Extract weighted avg metrics
    w_precision = cr['weighted avg']['precision']
    w_recall = cr['weighted avg']['recall']
    w_f1 = cr['weighted avg']['f1-score']
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Weighted Precision': w_precision,
        'Weighted Recall': w_recall,
        'Weighted F1-score': w_f1
    })
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)
print("\nComparison Table:\n", results_df)

# Bar Chart for all metrics
metrics = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-score']
results_df.plot(x='Model', y=metrics, kind='bar', figsize=(10, 6))
plt.ylim(0, 1)
plt.title('Comparison of Classification Models')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# # 7. Compare Model Accuracies

# In[20]:


result_df = pd.DataFrame(results)  # Create DataFrame before plotting

print("\nModel Comparison Table:\n", result_df)

plt.figure(figsize=(8,6))
bars = plt.bar(result_df['Model'], result_df['Accuracy'], color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


# # 8. Feature Importance for Tree-based Models

# In[21]:


import matplotlib.pyplot as plt


# In[22]:


for name, model in models.items():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(8,4))
        plt.barh(features, importances)
        plt.title(f"{name} Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()

