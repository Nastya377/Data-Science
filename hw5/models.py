import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def load_data_csv(file_path):
    try:
        csv_file = pd.read_csv(file_path)
        print("Данные успешно загружены.")
        return csv_file
    except FileNotFoundError as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    except TypeError as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)

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

# заполнение пустых ячеек  
def null_data_fill(null_data, type_fill = 'mean'):
    try:
        if type(null_data[0]) == str:
            null_data.fillna(null_data[0], inplace=True)
        elif type_fill == 'mean':
            null_data.fillna(null_data.mean(), inplace=True)
        elif type_fill == 'median':
            null_data.fillna(null_data.median, inplace=True)
        else: null_data.fillna(null_data[0], inplace=True)
    except TypeError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        print(f"Ошибка при расчетах: {e}")
        return np.array([])            
    except Exception as e:
        print(f"Ошибка при заполнении: {e}")
        return np.array([])