from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

# Load the dataset
file_path = "weather.csv"
df = pd.read_csv(file_path)

# Drop the date column
df = df.drop(columns=["date"])

# Encode the categorical target variable
label_encoder = LabelEncoder()
df["weather"] = label_encoder.fit_transform(df["weather"])

# Split data into features and target
X = df.drop(columns=["weather"])
y = df["weather"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and preprocessing objects
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        precipitation = float(request.form['precipitation'])
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])
        
        # Prepare input for model
        features = np.array([[precipitation, temp_max, temp_min, wind]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_weather = label_encoder.inverse_transform(prediction)[0]

        return render_template('predict.html', prediction=predicted_weather)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)