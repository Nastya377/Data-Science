import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.svm import SVC



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

   
       
    
    
    
    
    
    