import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from scipy import interpolate
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing, metrics
from sklearn.linear_model import LinearRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv("te1st_well.csv")
df.head()


print(df["LIQ_RATE"].count())
df.describe()
df = df.drop(["EXT_DATA", "PED_T", "U_OTP"], axis=1)
# Преобразование столбца DT_HOUR к типу datetime
df["DT_HOUR"] = pd.to_datetime(df['DT_HOUR'], format='%Y-%m-%d %H:%M:%S')
df["date"] = df['DT_HOUR'].apply(lambda x: x.strftime('%Y-%m-%d'))
df = df.groupby("date").agg("mean")
df = df.drop(["DT_HOUR"], axis=1)



# Создание экземпляра класса LinearRegression
model = LinearRegression()

# Генерация некоторых синтетических данных для демонстрации
X = np.array([['FREQ_HZ'], ['ACTIV_POWER'], ['PED_T']])  # Независимая переменная (признак)
y = np.array(["LIQ_RATE"])  # Зависимая переменная (целевая переменная)

# Обучение модели на тренировочных данных
model.fit(X, y)

# Вывод коэффициентов регрессии и свободного члена
print("Coef:", model.coef_)
print("Intercept:", model.intercept_)


df = df.dropna(subset=['LIQ_RATE'])
print(df)
df = df[["ACTIV_POWER", "LIQ_RATE", "PINP", "QGAS", "PLIN"]]
df = df.dropna()
print(df)

X = df[["ACTIV_POWER"]]
Y = df["LIQ_RATE"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.count())

model = LinearRegression().fit(X_train, Y_train)
p = model.__dict__
r_sq = model.score(X_train, Y_train)
slope = model.coef_
intercept = model.intercept_
print('coefficient of determination:', r_sq)
print('slope:', slope[0])
print('intercept:', intercept)

Y_predicted = model.predict(X_test)


print(plt.scatter(X_test, Y_test, color="green"))
plt.plot(X_test, Y_predicted, color="red")
plt.show()
