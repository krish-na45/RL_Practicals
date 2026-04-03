import pandas as pd
import numpy as np
url = "Iris.csv" 
df = pd.read_csv(url) 
 
print("First 5 rows of the dataset:") 
display(df.head()) 
 
print("\nDataset Information:") 
df.info() 
 
print("\nDataset Shape (rows, columns):") 
print(df.shape) 
 
print("\nColumn Names:") 
print(df.columns) 
 
print("\nMissing Values in Each Column:") 
print(df.isnull().sum()) 
 
print("\nStatistical Summary:") 
display(df.describe())
