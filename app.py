from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load the trained model from the .pkl file
model_file_path = 'random_forest_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Load the label encoders from .pkl files
with open('label_encoder_source.pkl', 'rb') as file:
    le_source = pickle.load(file)
    
with open('label_encoder_destination.pkl', 'rb') as file:
    le_destination = pickle.load(file)

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['No.', 'Info'])

# Encode categorical variables
label_encoders = {}
for column in ['Source', 'Destination']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Encode the target variable (Protocol)
le_protocol = LabelEncoder()
data['Protocol'] = le_protocol.fit_transform(data['Protocol'])

@app.route('/')
def index():
    return render_template('iindex.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process the file
        new_data = pd.read_csv(file_path)
        new_data = new_data.drop(columns=['No.', 'Info'])

        # Store the original Source and Destination columns
        original_source = new_data['Source'].copy()
        original_destination = new_data['Destination'].copy()

        def safe_encode(encoder, column):
            try:
                return encoder.transform(column)
            except:
                return [-1] * len(column)

        new_data['Source'] = safe_encode(le_source, new_data['Source'])
        new_data['Destination'] = safe_encode(le_destination, new_data['Destination'])

        features = new_data.drop(columns=['Protocol'], errors='ignore')
        probabilities = model.predict_proba(features)
        predicted_class = model.predict(features)
        max_probabilities = probabilities.max(axis=1)

        prob_threshold = 0.8
        filtered_indices = max_probabilities >= prob_threshold
        filtered_data = new_data[filtered_indices].copy()
        
        
        new_data = new_data.drop(columns=['Source', 'Destination'])
        # Add the original Source and Destination back to the filtered data
        filtered_data['Source'] = original_source.iloc[filtered_indices].values
        filtered_data['Destination'] = original_destination.iloc[filtered_indices].values
        #filtered_data['Original_Protocol'] = new_data['Protocol'].iloc[filtered_indices].values
        filtered_data['Predicted_Protocol'] = le_protocol.inverse_transform(predicted_class[filtered_indices].astype(int))
        filtered_data['Max_Probability'] = max_probabilities[filtered_indices]
 
        output_file_path = 'results/predicted_data.csv'
        filtered_data.to_csv(output_file_path, index=False)
        
        table_html = filtered_data.to_html(classes='table table-striped', index=False)
        
        return render_template('result.html', table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)