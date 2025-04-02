import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def Filter_Methods(X, y):
    try:
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Нормализация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Создание DataFrame из масштабированных данных для удобства
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

        # Фильтрационные Методы (Filter Methods)
        # Корреляционная Матрица
        corr_matrix = X_train_scaled_df.corr()

        # Визуализация корреляционной матрицы
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Корреляционная Матрица Признаков')
        plt.show()

        # Анализ корреляционной матрицы
        print("\nКорреляционная Матрица:")
        print(corr_matrix)

        # Определение порога корреляции для выявления высококоррелированных признаков
        threshold = 0.75

        # Поиск пар признаков с высокой корреляцией
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        if high_corr_pairs == []:
            print(f"\nНет признаков выше {threshold}")
        else:
            print(f"\nПары признаков с корреляцией выше {threshold}:")
            for pair in high_corr_pairs:
                print(pair)

    
    except Exception as e:
        logger.exception(f"Ошибка: {e}")


        
def feature_importance (X, y, df):
    try:
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

        # Модель случайного леса
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
    
      
        # Вывод важности признаков
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Важность признаков:")
        for idx in indices:
            print(f"Признак {idx}, важность: {importances[idx]}")

        # Визуализация важности признаков
        cols = [col for col in df.columns]
        plt.figure(figsize=(12,6))
        plt.title("Важность признаков")
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), np.array(cols)[indices], rotation=90)
        plt.xlabel('Признаки')
        plt.ylabel('Важность')
        plt.tight_layout()
        plt.savefig('img/feature_importance.png')
        plt.show()


    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        
        
        
        
def Quartile_Method(res_col, data_, type_='after'):
    try:
        for col in res_col:
            # Вычисление квартилей и межквартильного размаха
                Q1 = np.percentile(data_[col], 25)
                Q3 = np.percentile(data_[col], 75)
                IQR = Q3 - Q1

                # Определение границ для выбросов
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Поиск выбросов
                outliers = np.where((data_[col] < lower_bound) | (data_[col] > upper_bound))

                # Визуализация данных и выбросов
                plt.figure(figsize=(10, 6))
                plt.plot(data_[col], 'bo', label='Данные')
                plt.plot(outliers[0], 'ro', label='Выбросы')
    
                median = data_[col].median()
                mean = data_[col].mean()
                std_dev = data_[col].std()
                plt.axhline(median, color="purple", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
                plt.axhline(mean, color="blue", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")

    
                plt.axhline(Q1, color='orange', linestyle='dashed', linewidth=2, label=f"Q1 (Первый квартиль): {Q1:.2f}")
                plt.axhline(Q3, color='green', linestyle='dashed', linewidth=2, label=f"Q3 (Третий квартиль): {Q3:.2f}")
                plt.axhline(lower_bound, color='red', linestyle='dotted', linewidth=2, label=f"Нижняя граница: {lower_bound:.2f}")
                plt.axhline(upper_bound, color='red', linestyle='dotted', linewidth=2, label=f"Верхняя граница: {upper_bound:.2f}")
            
                plt.title('Обнаружение выбросов с использованием метода квартилей')
                plt.xlabel(col)
                plt.ylabel('Значение')
                plt.legend()
                plt.savefig(f"img/{type_}/Quartile Method_{col}.png")
                plt.show()

                # Вывод найденных выбросов
                print("Найденные выбросы:", outliers)        
        
    except Exception as e:
        logger.exception(f"Ошибка: {e}")       
        

        
def Sarima_model (data_ar):
    try:
        # Отключение предупреждений
        warnings.filterwarnings("ignore")

        # Разделение данных на обучающую и тестовую выборки
        train_size = int(len(data_ar) * 0.8)
        train_data, test_data = data_ar[:train_size], data_ar[train_size:]

        # SARIMA
        model = SARIMAX(train_data, order=(2, 1, 2), seasonal_order=(0, 2, 1, 12))
        model_fit = model.fit()


        # Визуализация исходных данных и сглаженных данных с использованием модели SARIMA
        plt.figure(figsize=(10, 6))
        plt.plot(train_data, label='Обучающие данные')
        plt.plot(model_fit.fittedvalues, color='red', label='Тестовые данные (SARIMA)')
        plt.title('Сравнение обучающих данных и тестовых данных')
        plt.legend()
        plt.show()

        # Прогнозирование для тестовой выборки
        forecast_test = model_fit.forecast(steps=len(test_data))

        # Определяем порядки p и q с помощью коррелограмм
        plot_pacf(data_ar)
        plt.show()

        # Визуализация прогноза на тестовой выборке
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(train_data)), train_data, label='Обучающие данные')
        plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), test_data, label='Тестовые данные')
        plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), forecast_test, color='green', label='Прогноз')
        plt.title('Прогноз на тестовой выборке (SARIMA)')
        plt.legend()
        plt.show()

        # Метрики качества
        mae = mean_absolute_error(test_data, forecast_test)
        mse = mean_squared_error(test_data, forecast_test)
        rmse = np.sqrt(mse)

        print(f'Mean Absolute Error (MAE): {mae:.3f}')
        print(f'Mean Squared Error (MSE): {mse:.3f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
        
        models_ = ['MAE', 'MSE', 'RMSE']
        accuracies = [mae, mse, rmse]

        plt.figure(figsize=(25, 20))
        sns.barplot(x=models_, y=accuracies, palette='viridis')
        plt.title(f"Метрики качества (SARIMA)")
        for index, value in enumerate(accuracies):
            plt.text(index, value + 0.02, f"{value:.2f}", ha='center')
        plt.savefig(f"img/Sarima_model_metrics.png")
        plt.show()

    except Exception as e:
        logger.exception(f"Ошибка: {e}")   
        
 

def Random_Forest_Regressor (data_ar, col):
    try:
        # Визуализируем временной ряд
        plt.figure(figsize=(12, 6))
        plt.plot(data_ar, label="Исходные данные")
        plt.title("Популярность")
        plt.xlabel("Даты")
        plt.ylabel("Популярность")
        plt.legend()
        plt.show()

        
        # Добавляем временные признаки (например, лаги), чтобы модель учитывала прошлые значения временного ряда.
        data_ar['Lag1'] = data_ar[col].shift(1)
        data_ar['Lag2'] = data_ar[col].shift(2)
        data_ar.dropna(inplace=True)

        # Разделение данных на обучающие и тестовые выборки
        train_size = int(len(data_ar) * 0.8)
        train_data = data_ar.iloc[:train_size]
        test_data = data_ar.iloc[train_size:]

        # Создание и обучение модели случайного леса с временными признаками
        model = RandomForestRegressor(n_estimators=100)
        model.fit(train_data[['Lag1', 'Lag2']], train_data[col])

        # Выполнение прогноза на тестовой выборке
        test_data['Predictions'] = model.predict(test_data[['Lag1', 'Lag2']])

        # Визуализация прогноза
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data[col], label='Обучающие данные')
        plt.plot(test_data.index, test_data[col], label='Тестовые данные')
        plt.plot(test_data.index, test_data['Predictions'], color='red', label='Прогноз')
        plt.title("Прогноз популярности фильмов с использованием случайного леса")
        plt.xlabel("Даты")
        plt.ylabel("Популярность")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Оценка точности прогноза
        mse = mean_squared_error(test_data[col], test_data['Predictions'])
        mae = mean_absolute_error(test_data[col], test_data['Predictions'])
        rmse = np.sqrt(mse)

        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")    
        
        models_ = ['MAE', 'MSE', 'RMSE']
        accuracies = [mae, mse, rmse]

        plt.figure(figsize=(25, 20))
        sns.barplot(x=models_, y=accuracies, palette='viridis')
        plt.title(f"Метрики качества (RandomForestRegressor)")
        for index, value in enumerate(accuracies):
            plt.text(index, value + 0.02, f"{value:.2f}", ha='center')
        plt.savefig(f"img/Random_Forest_Regressor_metrics.png")
        plt.show()
   
    except Exception as e:
        logger.exception(f"Ошибка: {e}")         
        
        
def Arima_model (data_ar, col):
    try:
        
        ts = data_ar
        
        # Визуализируем временной ряд
        plt.figure(figsize=(12, 6))
        plt.plot(ts, label="Исходные данные")
        plt.title("Популярность")
        plt.xlabel("Даты")
        plt.ylabel("Популярность")
        plt.legend()
        plt.show()

        # Проверка стационарности временного ряда с помощью теста Дики-Фуллера
        result = adfuller(ts)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] < 0.05:
            print("Ряд стационарен")
        else:
            print("Ряд нестационарен, необходимо дифференцирование")


        # Автоматический подбор параметров (p, d, q)
        model_auto = auto_arima(ts, order=(2, 1, 2), seasonal_order=(0, 2, 1, 12))
        print(model_auto.summary())

        # Разделение данных на обучающую и тестовую выборки
        train_size = int(len(ts) * 0.8)
        train_data, test_data = ts[:train_size], ts[train_size:]

        # Создаем и обучаем модель ARIMA на обучающих данных
        model_arima = ARIMA(train_data, order=model_auto.order)
        results_arima = model_arima.fit()

        # Прогнозирование на тестовой выборке
        forecast = results_arima.forecast(steps=len(test_data))

        # Визуализация прогноза
        plt.figure(figsize=(12, 6))
        plt.plot(train_data, label='Обучающие данные')
        plt.plot(test_data.index, test_data[col], label='Тестовые данные')
        plt.plot(test_data.index, forecast, color='red', linestyle='--', label='Прогноз (ARIMA)')
        plt.title("Прогноз популярности фильмов с использованием ARIMA")
        plt.xlabel("Даты")
        plt.ylabel("Популярность")
        plt.legend()
        plt.show()

        # Оценка точности прогноза
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)

        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        
        models_ = ['MAE', 'MSE', 'RMSE']
        accuracies = [mae, mse, rmse]

        plt.figure(figsize=(25, 20))
        sns.barplot(x=models_, y=accuracies, palette='viridis')
        plt.title(f"Метрики качества (ARIMA)")
        for index, value in enumerate(accuracies):
            plt.text(index, value + 0.02, f"{value:.2f}", ha='center')
        plt.savefig(f"img/Arima_model_metrics.png")
        plt.show()
        
    except Exception as e:
        logger.exception(f"Ошибка: {e}")         

        
     
def Voting_Regressor (X, y):
    try:     
        
        warnings.filterwarnings("ignore")
        
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

        # Определение ансамбля регрессоров (VotingRegressor)
        voting_regressor = VotingRegressor(estimators=[
            ('LinearRegression', LinearRegression()),
            ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=100)),
            ('RandomForestRegressor', RandomForestRegressor(n_estimators=100, random_state=100)),
            ('XGBRegressor', XGBRegressor()),
            ('Lasso', Lasso(alpha=0.1)),
            ('BayesianRidge', BayesianRidge()),
            ('GradientBoostingRegressor', GradientBoostingRegressor()),
            ('LGBMRegressor', LGBMRegressor()),
            ('CatBoostRegressor', CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=100, verbose=0)),
        ])


        # Обучение ансамбля регрессоров
        voting_regressor.fit(X_train, y_train)

        # Предсказание на тестовой выборке
        y_pred = voting_regressor.predict(X_test)

        # Вычисление метрик
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\nМетрики качества:\n")
        print(f'Mean Absolute Error (MAE): {mae:.3f}')
        print(f'Mean Squared Error (MSE): {mse:.3f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
        print(f'R²: {r2:.3f}')
        
        models_ = ['MAE', 'MSE', 'RMSE', 'R²']
        accuracies = [mae, mse, rmse, r2]

        plt.figure(figsize=(25, 20))
        sns.barplot(x=models_, y=accuracies, palette='viridis')
        plt.title(f"Метрики качества")
        for index, value in enumerate(accuracies):
            plt.text(index, value + 0.02, f"{value:.2f}", ha='center')
        plt.savefig(f"img/Voting_Regressor.png")
        plt.show()

        # Визуализация предсказанных и фактических значений
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
        plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
        plt.xlabel('Наблюдение')
        plt.ylabel('Значение')
        plt.title('Фактические и предсказанные значения')
        plt.legend()
        plt.show()        
    except Exception as e:
        logger.exception(f"Ошибка: {e}")          
        
        
        
        
        
        
        
        
        
        
        
        
        