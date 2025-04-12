import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\jhaas\Downloads\synthetic_fraud_dataset.csv')
df.head()
missing = df.isnull().sum()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
       desc = df.describe()
missing, desc
import seaborn as sns
import matplotlib.pyplot as plt

# Fraud distribution
sns.countplot(data=df, x='Fraud_Label')
plt.title("Fraudulent vs Non-Fraudulent Transactions")
plt.show()
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
sns.boxplot(x='Fraud_Label', y='Risk_Score', data=df)
plt.title("Risk Score vs Fraud Label")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
         X = df.drop(columns=['Transaction_ID', 'User_ID', 'Timestamp', 'Location',     'Fraud_Label'])
X = pd.get_dummies(X) 
y = df['Fraud_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
       # Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day_name()

sns.countplot(data=df[df['Fraud_Label'] == 1], x='Hour')
plt.title("Fraudulent Transactions by Hour")
plt.show()
sns.countplot(data=df[df['Fraud_Label'] == 1], x='Day', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title("Fraudulent Transactions by Day")
plt.xticks(rotation=45)
plt.show()
print("High-risk Devices:\n", df[df['Fraud_Label'] == 1]['Device_Type'].value_counts(normalize=True))
print("\nHigh-risk Card Types:\n", df[df['Fraud_Label'] == 1]['Card_Type'].value_counts(normalize=True))
print("Average Distance:\n", df.groupby('Fraud_Label')['Transaction_Distance'].mean())
fraud_ratio = df['Fraud_Label'].value_counts(normalize=True)
print(fraud_ratio)
sns.boxplot(x='Fraud_Label', y='Transaction_Amount', data=df)
plt.title("Transaction Amount vs Fraud")
plt.show()
print(df.groupby('Fraud_Label')['Transaction_Amount'].describe())
sns.histplot(data=df, x='Risk_Score', hue='Fraud_Label', kde=True, bins=30)
plt.title("Risk Score Distribution")
plt.show()
sns.kdeplot(data=df, x='Transaction_Distance', hue='Fraud_Label', common_norm=False)
plt.title("Distance of Transactions")
plt.show()
sns.countplot(data=df, x='Device_Type', hue='Fraud_Label')
plt.title("Fraud by Device Type")
plt.xticks(rotation=45)
plt.show()
sns.countplot(data=df, x='Card_Type', hue='Fraud_Label')
plt.title("Fraud by Card Type")
plt.show()
sns.countplot(data=df, x='Authentication_Method', hue='Fraud_Label')
plt.title("Fraud by Authentication Method")
plt.xticks(rotation=45)
plt.show()
top_states = df['Location'].value_counts().head(10).index
sns.histplot(data=df[df['Location'].isin(top_states)], x='Location', hue='Fraud_Label')
plt.title("Fraud by Top 10 States")
plt.xticks(rotation=45)
plt.show()
fraud_users = df[df['Fraud_Label'] == 1]['User_ID'].value_counts()
print(fraud_users[fraud_users > 1])  
fraud=df[df.Fraud_Label==1]['Transaction_Distance'].mean()
print(fraud)
normal_users = df[df['Fraud_Label'] == 0].groupby('User_ID')[['Transaction_Amount', 'Transaction_Distance']].mean()
fraud_users = df[df['Fraud_Label'] == 1].groupby('User_ID')[['Transaction_Amount', 'Transaction_Distance']].mean()
deviation = (fraud_users - normal_users).dropna()
print(deviation.head())
repeat_frauds = df[df['Fraud_Label'] == 1]['User_ID'].value_counts()
print("Repeat fraudsters:\n", repeat_frauds[repeat_frauds > 1])
spoof_check = df.groupby(['User_ID', 'Fraud_Label'])['Device_Type'].nunique().unstack()
print("Users with suspicious device variation:\n", spoof_check[spoof_check[1] > 1])
sns.lineplot(data=df, x='Transaction_Amount', y='Fraud_Label', ci=None)
plt.title("Fraud Probability by Transaction Amount")
plt.show()
def get_severity(row):
    score = row['Risk_Score'] + row['Transaction_Amount']*0.1 + row['Transaction_Distance']*0.2
    return 'High' if score > 80 else 'Medium' if score > 50 else 'Low'

df['Fraud_Severity'] = df[df['Fraud_Label']==1].apply(get_severity, axis=1)
print(df[['User_ID', 'Transaction_Amount', 'Fraud_Severity']].head())
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df['Week'] = df['Timestamp'].dt.to_period('W')
weekly_fraud = df.groupby('Week')['Fraud_Label'].mean()
weekly_fraud.plot(title="Weekly Fraud Rate Trend", marker='o')
plt.show()
from mlxtend.frequent_patterns import apriori, association_rules
cols = ['Merchant_Category', 'Card_Type', 'Transaction_Type']
df_encoded = pd.get_dummies(df[cols])

frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
import networkx as nx
import matplotlib.pyplot as plt
B = nx.Graph()
users = df['User_ID'].unique()
