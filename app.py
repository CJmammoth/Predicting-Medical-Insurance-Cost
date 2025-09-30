import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_and_train():
    data = pd.read_csv('insurance.csv')
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    scaler = StandardScaler()
    data[['age', 'bmi']] = scaler.fit_transform(data[['age', 'bmi']])
    X = data.drop('charges', axis=1)
    y = data['charges']
    model = LinearRegression()
    model.fit(X, y)
    return model, scaler, X.columns

model, scaler, feature_columns = load_and_train()



def main_app():
    st.title('Medical Insurance Cost Prediction App')
    st.write('This app predicts medical insurance costs based on personal data.')
    st.write('How old are you?')
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    st.write('What is your BMI?')
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    st.write('How many children do you have?')
    children = st.number_input('Children', min_value=0, max_value=10, value=0)
    st.write('Are you a smoker?')
    smoker = st.selectbox('Smoker', options=['yes', 'no'])
    st.write('Are you a male or a female?')
    gender = st.selectbox('Gender', options=['male', 'female'])
    st.write('Which region do you live in?')
    region = st.selectbox('Region', options=['northeast', 'northwest', 'southeast', 'southwest'])

    input_dict = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'sex_male' : 1 if gender == 'male' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }

    input_df = pd.DataFrame([input_dict], columns=feature_columns)
    input_df[['age', 'bmi']] = scaler.transform(input_df[['age', 'bmi']])

    if st.button('Predict Medical Cost'):
        prediction = model.predict(input_df)
        st.write(f'Estimated Medical Cost: ${prediction[0]:.2f}')

if __name__ == '__main__':
    main_app()
