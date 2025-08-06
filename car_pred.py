# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

#data loading and preprocessing
data = pd.read_csv("Car_Price_Prediction.csv")
data = pd.get_dummies(data, columns=["Year", "Model", "Transmission", "Fuel Type", ], drop_first=True)

x = data.drop(columns=["Price", "Make" ])
y = data["Price"]

#Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.2)

#Applying model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Model evaluation
print('coefficients:', model.coef_)
print('intercept:', model.intercept_)
print('mse:', mean_squared_error(y_test, y_pred))
print('r2 score:', r2_score(y_test, y_pred))

#visualization
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

#feature importance
feature_names = x.columns  
importances = model.coef_  

importance_data = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(importances) 
})

importance_data['base_feature'] = importance_data['feature'].apply(lambda x: x.split('_')[0])

grouped = importance_data.groupby('base_feature').sum().reset_index()

grouped = grouped.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='base_feature', data=grouped, palette='viridis')
plt.title('Grouped Feature Importances (Original Features)')
plt.xlabel('Total Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

#Cross-validation
model = LinearRegression()
scores = cross_val_score(model, x, y, cv=5, scoring='r2')
print("Cross-validated R² scores:", scores)
print("Average R²:", np.mean(scores))
