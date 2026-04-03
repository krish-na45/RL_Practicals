import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv("Iris.csv") 
sns.set(style="whitegrid") 
 
plt.figure(figsize=(5,4)) 
plt.hist(df['SepalLengthCm'], bins=10, color='skyblue', edgecolor='black') 
plt.title("Histogram of Sepal Length") 
plt.xlabel("Sepal Length (cm)") 
plt.ylabel("Frequency") 
plt.show() 
 
plt.figure(figsize=(5,4)) 
sns.boxplot(y=df['PetalLengthCm']) 
plt.title("Box Plot of Petal Length") 
plt.ylabel("Petal Length (cm)") 
plt.show() 
 
plt.figure(figsize=(5,4)) 
sns.scatterplot( x='SepalLengthCm',  y='PetalLengthCm', hue='Species', 
data=df ) 
plt.title("Scatter Plot: Sepal Length vs Petal Length") 
plt.xlabel("Sepal Length (cm)") 
plt.ylabel("Petal Length (cm)") 
plt.show()

plt.figure(figsize=(5,4)) 
sns.countplot(x='Species', data=df) 
plt.title("Bar Chart of Iris Species Count") 
plt.xlabel("Species") 
plt.ylabel("Count") 
plt.show() 
 
plt.figure(figsize=(6,5)) 
sns.heatmap( 
    df.drop(columns=['Id', 'Species']).corr(), 
    annot=True, 
    cmap='coolwarm' ) 
plt.title("Correlation Heatmap") 
plt.show() 
