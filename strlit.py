import streamlit as st
import random
from catboost import CatBoostClassifier # CatBoost

model = CatBoostClassifier()
model.load_model(r"catboost_model/model.json", "json")

#Caching the model for faster loading
@st.cache

def predict(creditScore, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember, estimatedSalary, country, gender):
    if hasCrCard == "Yes":
        hasCrCard = 1
    else:
        hasCrCard = 0

    if isActiveMember == "Yes":
        isActiveMember = 1
    else:
        isActiveMember = 0

    if country == 'Germany':
        germany = 1
        spain = 0
    elif country == 'Spain':
        germany = 0
        spain = 1
    else:
        germany = 0
        spain = 0

    if gender == 'Male':
        male = 1
    else:
        male = 0

    prediction = model.predict([[creditScore, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember, estimatedSalary, germany, spain, male]])
    return prediction

st.title('Bank Churn Prediction\nAccuracy: 0.903516 Precision: 0.930129 Recall:	0.873016 F1: 0.900668')
st.image("""https://storage.googleapis.com/kaggle-datasets-images/156197/358170/bd82bdbe49bd0bc65e26932fe58a3701/dataset-cover.jpg?t=2019-04-03-22-25-43""")
st.header('Enter the required characteristics below:')

opt1 = ['Yes', 'No']
opt2 = ['Yes', 'No']
opt3 = ['France', 'Germany', 'Spain']
opt4 = ['Male', 'Female']

if st.button('Randomize data'):
    st.session_state.creditScoreNI = random.randint(0, 1000)
    st.session_state.ageNI = random.randint(14, 150)
    st.session_state.tenureNI = random.randint(0, 20)
    st.session_state.balanceNI = random.uniform(0.0, 200000.0)
    st.session_state.numOfProductsNI = random.randint(1, 4)
    st.session_state.hasCrCardSB = opt1[random.randint(0, 1)]
    st.session_state.isActiveMemberSB = opt2[random.randint(0, 1)]
    st.session_state.estimatedSalaryNI = random.uniform(0.0, 200000.0)
    st.session_state.countrySB = opt3[random.randint(0, 2)]
    st.session_state.genderSB = opt4[random.randint(0, 1)]

creditScore = st.number_input('Credit Score:', min_value=0, max_value=1000, value=10, key='creditScoreNI')
age = st.number_input('Age:', min_value=14, max_value=150, value=14, key='ageNI')
tenure = st.number_input('Tenure:', min_value=0, max_value=20, value=1, key='tenureNI')
balance = st.number_input('Balance:', min_value=0.0, value=0.0, key='balanceNI')
numOfProducts = st.number_input('Number of products:', min_value=1, max_value=4, value=1, key='numOfProductsNI')
hasCrCard = st.selectbox('Has Credit Card?', opt1, key='hasCrCardSB')
isActiveMember = st.selectbox('Is active member?', opt2, key='isActiveMemberSB')
estimatedSalary = st.number_input('Estimated salary:', min_value=0.0, value=100.0, key='estimatedSalaryNI')
country = st.selectbox('Country:', opt3, key='countrySB')
gender = st.selectbox('Gender:', opt4, key='genderSB')


if st.button('Predict Client Exit'):
    pred = predict(creditScore, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember, estimatedSalary, country, gender)
    if pred > 0.5:
        exited = 'Yes'
    else:
        exited = 'No'
    st.success('Will the client exit? {}'.format(exited))
