
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def predictProba(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia,kc,pr_en,fats,carb,smoke):
    data = np.array([[sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia,kc,pr_en,fats,carb,smoke]])
    return model.predict_proba(data)

def predictDisease(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia,kc,pr_en,fats,carb,smoke):
    data = np.array([[sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia,kc,pr_en,fats,carb,smoke]])
    return model.predict(data)

def load_model():
    train = pd.read_excel("imt_disease_done.xlsx")
    train['Заболевание'] = train['Заболевание'].fillna('Не выявлено')
    disease =  train['Заболевание']
    pd.get_dummies(disease, prefix="Заболевание")
    train['Заболевание'].astype(str)
    labelencoder = LabelEncoder()
    train['Заболевание_переменная'] = pd.factorize(train['Заболевание'])[0]

    print("\nДанные с числовой целевой переменной:")
    train['Заболевание_переменная'].value_counts()

    class_imt = train['Классификация']
    pd.get_dummies(class_imt , prefix="Классификация")

    X = train.drop(['Заболевание','Классификация','Заболевание_переменная'], axis=1)
    y = train['Заболевание_переменная']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    return model

model = load_model()
#model.predict(X_test) #предсказываем на основе данных и фич параметры заболевания

def calculate_bmi(weight, height):
    if pd.isna(weight) or pd.isna(height):
        return None
    try:
        return round(weight / (height / 100) ** 2, 2)
    except ZeroDivisionError:
        print(f"Error: Height is 0 for row {weight}, {height}")
        return None

def classify_bmi(bmi_value):
    if bmi_value is None:
        return 'Undefined'
    elif bmi_value < 18.5:
        return 'Inadequate'
    elif bmi_value < 25:
        return 'Нормальный'
    elif bmi_value < 30:
        return 'Overweight'
    elif bmi_value < 35:
        return 'Obesity stage I'
    elif bmi_value < 40:
        return 'Obesity stage II'
    else:
        return 'Obesity stage III'

st.title('Prediction of hypertension, coronary heart disease and atherosclerosis')


st.subheader("Calculate user's BMI")

sex = st.number_input('Sex')
age = st.number_input('Age')
bmi = 0
imt = ""

# Ввод данных пользователя
col1, col2 = st.columns(2)
with col1:
    weight_input = st.number_input('Вес (кг)', min_value=0.0, value=70.0, step=0.1)
with col2:
    height_input = st.number_input('Рост (см)', min_value=50.0, max_value=250.0, value=170.0, step=1.0)


# Кнопка для расчета ИМТ
calculate_button = st.button('Рассчитать ИМТ')

if calculate_button:
    bmi_result = calculate_bmi(weight_input, height_input)

    if bmi_result is None:
        st.error("Failed to calculate BMI. Please check your entered data.")
    else:
        st.write(f"Your BMI: {bmi_result:.2f}")

    # Отображаем классификацию по ИМТ
    classification = classify_bmi(bmi_result)
    imt = classification
    bmi = bmi_result
    st.write(f"Classification: {classification}")


st.subheader("Enter user analysis parameters")

alt = st.number_input('ALT, 1/l')
ast = st.number_input('AST, 1/l')
bil_ob = st.number_input('Total bilirubin, mmol/l')
bil_pr = st.number_input('Direct (conjugated) bilirubin, mmol/l')
glucose = st.number_input('Glucose, mmol/l')
kreatinin = st.number_input('Creatinine, mmol/l')
lpnp = st.number_input('LDL, mmol/l')
lpvp = st.number_input('HDL, mmol/l')
protein = st.number_input('Total protein, g/l')
tgr = st.number_input('Triglycerides, mmol/l')
holesterin = st.number_input('Total cholesterol, mmol/l')
ia = st.number_input('Atherogenicity index')

st.subheader("Enter user nutrition information")

kc = st.number_input('Energy, kcal')
pr_en = st.number_input('Proteins, g')
fats = st.number_input('Fats, g')
carb = st.number_input('Carbohydrates, g')
smoke = st.number_input('Do you use tobacco products?')

done = st.button('Calculate the risks of disease')
# Кнопка для получения прогноза

if done:
    result = predictProba(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia,kc,pr_en,fats,carb,smoke)
    str = "The risk of hypertension is "+ str(result[0][0] * 100)  +  "%\n" + "The risk of coronary heart disease is "+ str(result[0][1] * 100)  +  "%\n" +"The risk of atherosclerosis is equal to "+ str(result[0][2] * 100) +  "%\n" #  +"The patient is likely "+ str(result[0][3] * 100)  +  "% healthy\n"
    if result is None:
        st.error("Failed to calculate.")
    else:
        st.text(str)
