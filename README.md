Medical Insurance Cost Prediction App
This Streamlit web application predicts medical insurance costs based on user input such as age, BMI, number of children, smoking status, gender, and region. It uses a linear regression model trained on the insurance.csv dataset.
🚀 Features
- Interactive user interface built with Streamlit
- Real-time prediction of insurance charges
- Preprocessing with one-hot encoding and feature scaling
- Linear regression model trained on historical data
- Caching for efficient model loading

📦 Requirements
Install dependencies using pip:
-pip install pandas scikit-learn streamlit

🧠 How It Works
- Loads and preprocesses the dataset:
- One-hot encodes categorical features
- Scales numerical features (age, bmi)
- Trains a linear regression model
- Accepts user input via Streamlit widgets
- Transforms input to match training format
- Predicts insurance cost and displays result

Running the App
streamlit run app.py

📊 Model Details
- Algorithm: Linear Regression
- Target Variable: charges
- Features: Age, BMI, Children, Smoker, Gender, Region (encoded)

🛠️ Customization Ideas
- Swap Linear Regression for more advanced models (e.g., Random Forest)
- Add model evaluation metrics (R², MAE)
- Save/load model with joblib or pickle
- Add input validation and error handling


Let me know if you want to add screenshots, deployment instructions, or a GitHub badge!

Contact: 
Carlos Jamito
carlosjamito@gmail.com
