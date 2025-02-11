import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# подсчет пустых ячеек
def null_data(null_data):
    try:
        return null_data.isnull().sum()
    except TypeError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])
    except Exception as e:
        print(f"Ошибка при подсчете: {e}")
        return np.array([])
    
# отчет по пустым ячеекам
def null_data_report(null_data):
    try:
        #null_data.info()
        print("Пропущенные значения: ", data[null_data.isna()], sep='\n')
        print(f"Доля пропущенных значений в данных: {null_data.isnull().sum()/len(null_data)}")
    except TypeError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])
    except Exception as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])

# заполнение пустых ячеек  
def null_data_fill(null_data):
    try:
        if type(null_data[0]) == str:
            null_data.fillna(null_data[0], inplace=True)
        else:null_data.fillna(null_data.mean(), inplace=True)
    except TypeError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])            
    except Exception as e:
        print(f"Ошибка при заполнении: {e}")
        return np.array([])

    
 # Визуализация
def scatter_data(plot_data, num_points=90):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_points), plot_data[:num_points], color='blue', label='Сумма ЗП')
    plt.title(f'Общая сумма выплаченной зарплаты в USD (первые {num_points} точек)')
    plt.legend()
    plt.show()

def hist_data(plot_data):
    plt.figure(figsize=(10, 6))
    plt.hist(plot_data, bins=20, density=True, color='blue', label='Сумма ЗП')
    plt.title(f'Общая сумма выплаченной зарплаты в USD')
    plt.legend()
    plt.show()
    
def bar_data(plot_data_x, plot_data_y):
    plt.figure(figsize=(10, 6))
    plt.bar(plot_data_x, plot_data_y)
    plt.title(f'Общая сумма выплаченной зарплаты в USD')
    plt.legend()
    plt.show()

def plot_data(plot_data, num_points=100):
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data[:num_points])
    plt.title(f'Общая сумма выплаченной зарплаты в USD(первые {num_points} точек)')
    plt.legend()
    plt.show()      
