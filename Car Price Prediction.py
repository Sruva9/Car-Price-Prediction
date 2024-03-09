import pandas as pd  # For data analysis and manipulation
import numpy as np  # For mathematical operations and data handling
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # Data visualization library based on Matplotlib

# Read the dataset from a CSV file
df = pd.read_csv('car_data.csv')

# Display the first 5 rows of the dataset
df.head(5)

# Create a heatmap to visualize the correlation between numerical columns
plt.figure(figsize=(9, 9))
sns.heatmap(df.corr())
plt.show()

# Create a distribution plot for the 'price' column
plt.figure(figsize=(10, 10))
sns.distplot(df.Present_Price)
plt.show()

# Extract specific columns from the dataset
data = pd.DataFrame(df)
y = data['Present_Price']
x = data['Car_Name']

# Create a bar plot for 'car name' vs. 'price'
pr = plt.figure(figsize=(40, 15))
plt.xlabel("Car Name", fontsize=40)
plt.ylabel("Price of Car", fontsize=40)
plt.xticks(fontweight='bold', rotation='horizontal', fontsize=20)
plt.yticks(fontweight='bold', rotation='horizontal', fontsize=20)
plt.bar(x, y, color='b', align='center')
plt.show()

# Create a bar plot for 'Driven Km' vs. 'price'
plt.figure(figsize=(50, 10))
sns.barplot(data=df, x='Driven_kms', y='Present_Price')
plt.xlabel("Driven_kms", fontsize=40)
plt.ylabel("Price of Car", fontsize=40)

# Extract more columns from the dataset
data = pd.DataFrame(df)
y = data['Present_Price']
x = data['Selling_Price']
x1 = data['Fuel_Type']

# Create a bar plot for 'Selling Price' vs. ' Present Price'
pr = plt.figure(figsize=(40, 15))
plt.xlabel("Selling Price", fontsize=40)
plt.ylabel("Price of Car", fontsize=40)
plt.xticks(fontweight='bold', rotation='horizontal', fontsize=20)
plt.yticks(fontweight='bold', rotation='horizontal', fontsize=20)
plt.bar(x, y, color='b', align='center')
plt.show()

# Create a bar plot for 'cylindernumber' vs. 'price'
y = data['Present_Price']
x1 = data['Fuel_Type']
pr = plt.figure(figsize=(40, 15))
plt.xlabel("Fuel Type", fontsize=40)
plt.ylabel("Price of Car", fontsize=40)
plt.xticks(fontweight='bold', rotation='horizontal', fontsize=20)
plt.yticks(fontweight='bold', rotation='horizontal', fontsize=20)
plt.bar(x1, y, color='r', align='center')
plt.show()

# Create a violin plot to visualize the relationship between 'price' and 'CarName' with different categories
sns.violinplot(data=df, y='Present_Price', hue='Car_Name')

# Create a scatter plot with regression line for 'horsepower' vs. 'price'
pr = plt.figure(figsize=(7, 5))
plt.title("Correlation between Driven_kms and Price")
sns.regplot(x="Driven_kms", y="Present_Price", data=df)

# Data preparation for machine learning
p = "Present_Price"
data = df[["Driven_kms", "Selling_Price", "Fuel_Type", "Present_Price"]]
x = np.array(data.drop([p], 1))
y = np.array(data[p])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a Decision Tree Regressor model and make predictions
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.score(X_test, predictions)

# Create a Linear Regression model and make predictions
from sklearn.linear_model import LinearRegression
m = LinearRegression()
m.fit(X_train, y_train)
prediction = m.predict(X_train)
model.score(X_test, predictions)