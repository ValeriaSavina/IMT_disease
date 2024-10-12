
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

def predictProba(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia):
    data = np.array([[sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia]])
    return model.predict_proba(data)

def predictDisease(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia):
    data = np.array([[sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia]])
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
        print(f"Ошибка: Рост равен 0 для строки {weight}, {height}")
        return None

def classify_bmi(bmi_value):
    if bmi_value is None:
        return 'Недопределено'
    elif bmi_value < 18.5:
        return 'Недостаточный'
    elif bmi_value < 25:
        return 'Нормальный'
    elif bmi_value < 30:
        return 'Избыточный вес'
    elif bmi_value < 35:
        return 'Ожирение I степени'
    elif bmi_value < 40:
        return 'Ожирение II степени'
    else:
        return 'Ожирение III степени'

st.title('Прогнозирование гипертонии, ИБС и атеросклероза')


st.subheader("Подсчёт ИМТ пациента")

sex = st.number_input('Пол')
age = st.number_input('Возраст')
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
        st.error("Не удалось рассчитать ИМТ. Проверьте введенные данные.")
    else:
        st.write(f"Ваш ИМТ: {bmi_result:.2f}")

    # Отображаем классификацию по ИМТ
    classification = classify_bmi(bmi_result)
    imt = classification
    bmi = bmi_result
    st.write(f"Классификация: {classification}")


st.subheader("Введите параметры анализов пациента")

alt = st.number_input('АЛТ, 1/л')
ast = st.number_input('АСТ, 1/л')
bil_ob = st.number_input('Билирубин общий, мкмоль/л')
bil_pr = st.number_input('Билирубин прямой (связанный), мкмоль/л')
glucose = st.number_input('Глюкоза, ммоль/л')
kreatinin = st.number_input('Креатинин, мкмоль/л')
lpnp = st.number_input('ЛПНП, ммоль/л')
lpvp = st.number_input('ЛПВП, ммоль/л')
protein = st.number_input('Белок общий, г/л')
tgr = st.number_input('Триглицериды, ммоль/л')
holesterin = st.number_input('Общий холестерин, ммоль/л')
ia = st.number_input('Индекс атерогенности')

done = st.button('Вычислить риски')
recomendation = st.button('Посмотреть рекомендации')
rec = 0
# Кнопка для получения прогноза

def printprof (disease):
    result = ""
    if disease == 0:
        result += """ 
        Профилактика гипертензии: 
        1. Увеличение физической и двигательной нагрузки.
        2. Отказ от курения: Табак и содержащийся в нем никотин способствуют спазму артерий. 
        3. Отказ от употребления алкоголя.
        4. Оптимизацию питания: ограничение суточного потребления поваренной соли, не досаливать пищу после готовки;
        Наибольшее значение имеют такие факторы, как злоупотребление кофе, шоколадом и крепким чаем, употребление большого количества животных жиров и простых углеводов, 
        регулярное потребление соленых блюд, переедание.
        5. Коррекцию массы тела: калорийность пищи не должна быть высокой. Требуется употреблять пищу небольшими порциями. 
        6. Соблюдение режима сна и бодрствования: ночной сон должен составлять 8-9 часов. 
        7. Исключение стрессовых состояний. Стресс является пусковым фактором многих заболеваний.
        8. В случае ощущений недомогания или ухудшения самочувствия, необходима консультация врача. """
    elif disease == 1:
        result +=  """
        Профилактика атеросклероза:
        1)Отказ от курения
        2)Отказ от употребления алкоголя
        3)Антиатеросклеротическая диета — например, «средиземноморская»: масло, богатое полиненасыщенными жирными кислотами (Омега-3): 
        льняное, рапсовое или оливковое. 
        Из алкоголя только столовое вино до 150 мл в день (но лучше полностью отказаться от алкоголя, так как он является фактором риска возникновения инсульта). 
        Отказ от хлеба из муки высших сортов, ни дня без фруктов и овощей.
        4)Активно-подвижный образ жизни — регулярные дозированные физические нагрузки.
        5)Поддержание психологического и физического комфорта
        6)Снижение и поддержание массы тела. """
    elif disease == 2:
        result += """
        Профилактика ИБС:
        1)правильное питание;
        2)снижение в крови концентрации липопротеидов низкой плотности;
        3)снижение в крови концентрации глюкозы;
        4)поддержание нормальной массы тела;
        5)снижение артериального давления;
        6)избавление от вредных привычек;
        7)повышение двигательной активности;
        8)поддержание оптимального соотношения труда и отдыха."""
    return result
    
if recomendation:
    st.text (printprof(rec))

if done:
    result = predictProba(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia)
    rec = predictDisease(sex,age,height_input,weight_input,bmi,alt,ast,bil_ob,bil_pr,glucose,kreatinin,lpnp,lpvp,protein,tgr,holesterin,ia)
    string = "Риск гипертонии равен "+ str(result[0][0] * 100)  +  "%\n" + "Риск ИБС равен "+ str(result[0][1] * 100)  +  "%\n" +"Риск атеросклероза равен "+ str(result[0][2] * 100) +  "%\n" #  +"Пациент с вероятностью "+ str(result[0][3] * 100)  +  "% здоров\n"
    if result is None:
        st.error("Не удалось рассчитать.")
    else:
        st.text(string)


