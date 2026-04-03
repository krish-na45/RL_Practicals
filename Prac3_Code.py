import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
 
df = pd.read_csv('/content/sample_data/california_housing_train.csv') 
print("Dataset loaded successfully. First 5 rows:") 
display(df.head()) 
 
X = df['total_rooms'].values.reshape(-1, 1) 
y = df['median_house_value'] 
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}") 
 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='total_rooms', y='median_house_value', data=df) 
plt.title('Scatter Plot of Total Rooms vs. Median House Value') 
plt.xlabel('Total Rooms') 
plt.ylabel('Median House Value') 
plt.grid(True)

plt.show() 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
print(f"Training data shape: X_train={X_train.shape}, 
y_train={y_train.shape}") 
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}") 
 
model = LinearRegression() 
model.fit(X_train, y_train) 
print("Model training complete.") 
 
y_pred = model.predict(X_test) 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
 
print(f"Mean Squared Error (MSE): {mse:.2f}") 
print(f"R-squared (R2) Score: {r2:.2f}") 
print(f"Model Intercept: {model.intercept_:.2f}") 
print(f"Model Coefficient (Slope): {model.coef_[0]:.2f}") 
 
plt.figure(figsize=(10, 6)) 
plt.scatter(X_test, y_test, label='Actual Values', alpha=0.6) 
plt.plot(X_test, y_pred, color='red', label='Regression Line') 
plt.title('Linear Regression: Total Rooms vs. Median House Value') 
plt.xlabel('Total Rooms') 
plt.ylabel('Median House Value') 
plt.legend() 
plt.grid(True) 
plt.show()
