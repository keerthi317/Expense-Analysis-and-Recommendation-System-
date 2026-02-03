import pandas as pd
df = pd.read_csv('Budget_Analysis.csv')
print(df.head())
print(df.tail())
print(df.duplicated().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
print(df.head())
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['amount'] >= lower_bound) & (df['amount'] <= upper_bound)]
print("\nAfter Outlier Removal")
print(df.shape)
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Month_Name'] = df['date'].dt.month_name()
df['Day'] = df['date'].dt.day
df['Day_Name'] = df['date'].dt.day_name()
df['Weekday'] = df['date'].dt.weekday
print("\nDate Features Added")
print(df[['date','Year','Month','Month_Name']].head())
import numpy as np
df['amount'] = df['amount'].abs()   # Ensure no negative values
df['Log_Amount'] = np.log1p(df['amount'])  # Helps ML models
df['High_Expense'] = df['amount'].apply(lambda x: 1 if x > df['amount'].mean() else 0)
df['Year_Month'] = df['date'].dt.to_period('M')
monthly_stats = df.groupby('Year_Month')['amount'].agg(
    Monthly_Total='sum',
    Monthly_Avg='mean',
    Monthly_Max='max'
)
df = df.merge(monthly_stats, on='Year_Month', how='left')

print("\nFinal Feature Engineered Dataset")
print("Shape:", df.shape)
print(df.head())
print("\nColumns:")
print(df.columns)
print("Feature Engineered Dataset Shape:", df.shape)
print(df.head())
print(df.columns)
# IMplementaion of Linear Regressing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
# Features & Target
X = df[['category', 'Month']]
y = df['amount']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Model
lr_model = LinearRegression()
print(lr_model.fit(X_train, y_train))
y_pred = lr_model.predict(X_test)
# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))
# Implementation of RandomForest Classifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Features & Target
X = df[['amount',  'Month']]
y = df['category']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
print(rf_model.fit(X_train, y_train))
# Prediction
y_pred = rf_model.predict(X_test)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:",classification_report(y_test, y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))