pip install torch torchvision torchaudio
######
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Define the model class 
class PredictiveModel(nn.Module):
    def __init__(self, input_dim):
        super(PredictiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Load the trained model
input_dim = 46  # for 46 features
model = PredictiveModel(input_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # evaluation mode

# Function to make predictions
def predict_dementia(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()

# Streamlit app
st.title("Dementia Prediction App")

# input prompt for user to enter data
st.header("Enter the patient data:")
input_data = []

# Binary features
input_data.append(1 if st.selectbox('Diabetic', ['Yes', 'No']) == 'Yes' else 0)


input_data.append(st.number_input('Alcohol Level', value=0.0))
input_data.append(st.number_input('Heart Rate', value=0.0))
input_data.append(st.number_input('Blood Oxygen Level', value=0.0))
input_data.append(st.number_input('Body Temperature', value=0.0))
input_data.append(st.number_input('Weight', value=0.0))
input_data.append(st.number_input('MRI Delay', value=0.0))
# One-hot encoded features
prescription = st.selectbox('Prescription', ['Donepezil', 'Galantamine', 'Memantine', 'None', 'Rivastigmine'])
input_data += [1 if prescription == 'Donepezil' else 0,
               1 if prescription == 'Galantamine' else 0,
               1 if prescription == 'Memantine' else 0,
               1 if prescription == 'None' else 0,
               1 if prescription == 'Rivastigmine' else 0]
input_data.append(st.number_input('Dosage in mg', value=0.0))
input_data.append(st.number_input('Age', value=0))
input_data.append(st.number_input('Cognitive Test Scores', value=0))


education_level = st.selectbox('Education Level', ['Diploma/Degree', 'No School', 'Primary School', 'Secondary School'])
input_data += [1 if education_level == 'Diploma/Degree' else 0,
               1 if education_level == 'No School' else 0,
               1 if education_level == 'Primary School' else 0,
               1 if education_level == 'Secondary School' else 0]

dominant_hand = st.selectbox('Dominant Hand', ['Left', 'Right'])
input_data += [1 if dominant_hand == 'Left' else 0,
               1 if dominant_hand == 'Right' else 0]

gender = st.selectbox('Gender', ['Female', 'Male'])
input_data += [1 if gender == 'Female' else 0,
               1 if gender == 'Male' else 0]

family_history = st.selectbox('Family History', ['No', 'Yes'])
input_data += [1 if family_history == 'No' else 0,
               1 if family_history == 'Yes' else 0]

smoking_status = st.selectbox('Smoking Status', ['Current Smoker', 'Former Smoker', 'Never Smoked'])
input_data += [1 if smoking_status == 'Current Smoker' else 0,
               1 if smoking_status == 'Former Smoker' else 0,
               1 if smoking_status == 'Never Smoked' else 0]

apoe_e4 = st.selectbox('APOE ε4', ['Negative', 'Positive'])
input_data += [1 if apoe_e4 == 'Negative' else 0,
               1 if apoe_e4 == 'Positive' else 0]

physical_activity = st.selectbox('Physical Activity', ['Mild Activity', 'Moderate Activity', 'Sedentary'])
input_data += [1 if physical_activity == 'Mild Activity' else 0,
               1 if physical_activity == 'Moderate Activity' else 0,
               1 if physical_activity == 'Sedentary' else 0]

depression_status = st.selectbox('Depression Status', ['No', 'Yes'])
input_data += [1 if depression_status == 'No' else 0,
               1 if depression_status == 'Yes' else 0]

medication_history = st.selectbox('Medication History', ['No', 'Yes'])
input_data += [1 if medication_history == 'No' else 0,
               1 if medication_history == 'Yes' else 0]

nutrition_diet = st.selectbox('Nutrition Diet', ['Balanced Diet', 'Low-Carb Diet', 'Mediterranean Diet'])
input_data += [1 if nutrition_diet == 'Balanced Diet' else 0,
               1 if nutrition_diet == 'Low-Carb Diet' else 0,
               1 if nutrition_diet == 'Mediterranean Diet' else 0]

sleep_quality = st.selectbox('Sleep Quality', ['Good', 'Poor'])
input_data += [1 if sleep_quality == 'Good' else 0,
               1 if sleep_quality == 'Poor' else 0]

chronic_health_conditions = st.selectbox('Chronic Health Conditions', ['Diabetes', 'Heart Disease', 'Hypertension', 'None'])
input_data += [1 if chronic_health_conditions == 'Diabetes' else 0,
               1 if chronic_health_conditions == 'Heart Disease' else 0,
               1 if chronic_health_conditions == 'Hypertension' else 0,
               1 if chronic_health_conditions == 'None' else 0]

# Predict button
if st.button("Predict"):
    
    input_data = [input_data]

    # Standardize the input data 
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    prediction = predict_dementia(input_data[0])
    st.write(f"Prediction probability: {prediction:.2f}")
    if prediction >= 0.5:
        st.write("The model predicts that the patient has a high risk of dementia.")
    else:
        st.write("The model predicts that the patient has a low risk of dementia.")
