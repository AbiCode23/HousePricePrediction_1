import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask, render_template, request
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Desktop\6sem\house_data.csv')

# Select relevant columns
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

# Split the dataset into features (X) and target variable (y)
X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(lr, open('model.pkl', 'wb'))

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']
    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))

if __name__ == '__main__':
    app.run(debug=True)
