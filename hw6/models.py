import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm
from scipy.stats import gaussian_kde
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_alltype(file_path, typefile = 'csv'):
    try:
        if typefile == 'csv':
            file_result = pd.read_csv(file_path)
            logging.info("Данные успешно загружены.")
        elif typefile == 'json':
            file_result = pd.read_json(file_path)
            logging.info("Данные успешно загружены.")
        elif typefile == 'excel':
            file_result = pd.read_excel(file_path)
            logging.info("Данные успешно загружены.")
        else:
            logging.info("Возможно выгрузить из фалов csv, json и excel.")
        return file_result
    except FileNotFoundError as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        return np.array([])
    except ValueError as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        return np.array([])
    except TypeError as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        return np.array([])
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        return np.array([])

    
def data_describe (data_info):
    try:
        data_info.head()
        print (f"Размер датасета: {len(data_info)}", sep='\n')
        print(f"Проверка наличия дубликатов: {data_info.duplicated().sum()}", sep='\n')
        print("Анализ пропущенных значений:", data_info.isnull().sum(), sep='\n')
    except TypeError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])
    except Exception as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])

    
def null_data_report (null_data):
    try:
        print ("Анализ пропущенных значений:")
        print("Пропущенные значения: ", null_data[null_data.isna()], sep='\n')
        print(f"Доля пропущенных значений в данных: {null_data.isnull().sum()/len(null_data)}", sep='\n')
    except TypeError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])
    except Exception as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])    
    
    
# заполнение пустых ячеек  
def null_data_fill(null_data, colum, type_fill = 'mean'):
    try:
        warnings.filterwarnings("ignore")
        if type(null_data[colum][0]) == str:
            null_data.loc[:, colum]=null_data.loc[:, colum].fillna(null_data[colum][0])
        elif type_fill == 'mean':
            null_data[colum].fillna(null_data[colum].mean(), inplace=True)
        elif type_fill == 'median':
            null_data[colum].fillna(null_data[colum].median, inplace=True)
        else: null_data[colum].fillna(null_data[colum][0])
    except TypeError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])            
    except Exception as e:
        logging.error(f"Ошибка при заполнении: {e}")
        return np.array([])    
    

# удаление дубликатов  
def duplicates_data_del (duplicates_data):
    try:
        duplicates_data.drop_duplicates()
    except TypeError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])
    except ValueError as e:
        logging.error(f"Ошибка при расчетах: {e}")
        return np.array([])            
    except Exception as e:
        logging.error(f"Ошибка при заполнении: {e}")
        return np.array([])

    
# Распределение целевой переменной
def distribution_of_variable (y):

    plt.figure(figsize=(10, 6))
    # Построение гистограммы с KDE
    ax = sns.histplot(
        y,
        kde=True,
        bins=30,
        edgecolor="black",
        color="royalblue",
        alpha=0.8,
        linewidth=1.2
    )

    # Изменение цвета KDE линии
    kde_color = "darkorange"
    sns.kdeplot(y, color=kde_color, linewidth=2, label="KDE (плотность)")

    plt.title(f"Гистограмма и KDE", fontsize=14, fontweight="bold")
    plt.xlabel("Рейтинг", fontsize=12)
    plt.ylabel("Плотность / Частота", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Аннотация медианы и среднего
    median = y.median()
    mean = y.mean()
    std_dev = y.std()
    plt.axvline(median, color="red", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
    plt.axvline(mean, color="blue", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")

            
    # Добавляем перцентильные линии
    percentiles = [0.25, 0.75, 0.99]
    percentile_values = y.quantile(percentiles)
    perc_colors = ["green", "purple", "brown"]  # Цвета для перцентилей
    for perc, value, color in zip(percentiles, percentile_values, perc_colors):
        plt.axvline(value, color=color, linestyle=":", linewidth=2, label=f"{int(perc * 100)}-й перцентиль: {value:.2f}")

    # Добавляем идеальное нормальное распределение

    x_range = np.linspace(y.min(), y.max(), 1000)  # Диапазон значений
    ideal_pdf = norm.pdf(x_range, loc=mean, scale=std_dev)  # Плотность вероятности нормального распределения
    ideal_pdf_scaled = ideal_pdf * len(y) * (y.max() - y.min()) / 30  # Масштабируем для совпадения с гистограммой
    plt.plot(x_range, ideal_pdf_scaled, color="orange", linestyle="--", linewidth=2.5, label="Идеальное распределение")


    kde = gaussian_kde(y)
    kde_values = kde(x_range)
    peak_x = x_range[np.argmax(kde_values)]
    peak_y = kde_values.max()

               
            # Легенда
    plt.legend(fontsize=10)
    plt.savefig('img/distribution_of_variable.png')
    plt.show()

    
# Корреляционная матрица
def corr_matrix (data_corr):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Матрица корреляции')
    plt.savefig('img/corr_matrix.png')
    plt.show()

    
# Анализ выбросов
def boxplot_analysis (data_analysis, col):
    plt.figure(figsize=(10, 8))
    data_analysis.boxplot(column = col)
    plt.title('Анализ выбросов')
    plt.savefig('img/boxplot.png')
    plt.show()    
   

def plot_kde_distributions(data_main, numeric_columns):
    summary_data = []

    for column in numeric_columns:
        data_1 = data_main[column].dropna()

        # Проверяем уникальность значений
        if data_1.nunique() <= 1:  # Если в данных только одно уникальное значение
            print(f"Пропущен график для столбца {column}, так как все значения одинаковы или отсутствует дисперсия.")
            continue

        plt.figure(figsize=(10, 6))
        # Построение гистограммы с KDE
        ax = sns.histplot(
            data_1,
            kde=True,
            bins=30,
            edgecolor="black",
            color="royalblue",
            alpha=0.8,
            linewidth=1.2
            )

        # Изменение цвета KDE линии
        sns.kdeplot(data_1, color="darkorange", linewidth=2, label="KDE (плотность)")
        plt.title(f"Гистограмма и KDE для признака: {column}", fontsize=14, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Плотность / Частота", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Аннотация медианы и среднего
        median = data_1.median()
        mean = data_1.mean()
        std_dev = data_1.std()
        plt.axvline(median, color="red", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
        plt.axvline(mean, color="blue", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")

        # Добавляем перцентильные линии
        percentiles = [0.25, 0.75, 0.99]
        percentile_values = data_1.quantile(percentiles)
        perc_colors = ["green", "purple", "brown"]  # Цвета для перцентилей
        for perc, value, color in zip(percentiles, percentile_values, perc_colors):
            plt.axvline(value, color=color, linestyle=":", linewidth=2, label=f"{int(perc * 100)}-й перцентиль: {value:.2f}")

        # Добавляем идеальное нормальное распределение
        try:
            x_range = np.linspace(data_1.min(), data_1.max(), 1000)  # Диапазон значений
            ideal_pdf = norm.pdf(x_range, loc=mean, scale=std_dev)  # Плотность вероятности нормального распределения
            ideal_pdf_scaled = ideal_pdf * len(data_1) * (data_1.max() - data_1.min()) / 30  # Масштабируем для совпадения с гистограммой
            plt.plot(x_range, ideal_pdf_scaled, color="orange", linestyle="--", linewidth=2.5, label="Идеальное распределение")
        except Exception as e:
            print(f"Ошибка при построении идеального распределения для столбца {column}: {e}")

        # Вычисляем пиковое значение KDE
        try:
            kde = gaussian_kde(data_1)
            kde_values = kde(x_range)
            peak_x = x_range[np.argmax(kde_values)]
            peak_y = kde_values.max()

            # Аннотация пика
            plt.annotate(f"Пик: {peak_x:.2f}", xy=(peak_x, peak_y),
                         xytext=(peak_x + 0.5, peak_y + 0.1),
                         arrowprops=dict(facecolor='black', arrowstyle="->"),
                         fontsize=10)
        except np.linalg.LinAlgError:
            print(f"Не удалось построить KDE для столбца {column}, так как данные имеют низкую дисперсию.")
            peak_x = None

        # Легенда
        plt.legend(fontsize=10)

        plt.show()
        plt.close()

        # Расчет статистик
        kurt = kurtosis(data_1)
        skewness = skew(data_1)

        # Определение распределения
        if -0.5 <= skewness <= 0.5:
            distribution = "Нормальное"
        elif skewness > 0.5:
            distribution = "Смещенное вправо"
        elif skewness < -0.5:
            distribution = "Смещенное влево"
        else:
            distribution = "Неопределено"

        # Интервал, в котором распределено большинство значений (межквартильный диапазон)
        lower, upper = data_1.quantile(0.25), data_1.quantile(0.75)
        range_info = f"[{lower:.2f}, {upper:.2f}]"

        # Сохранение данных в итоговую таблицу
        summary_data.append({
            "Признак": column,
            "Эксцесс": round(kurt, 2),
            "Асимметрия": round(skewness, 2),
            "Пик": round(peak_x, 2) if peak_x else None,
            "Среднее": round(mean, 2),
            "Медиана": round(median, 2),
            "25-й перцентиль": round(percentile_values[0.25], 2),
            "75-й перцентиль": round(percentile_values[0.75], 2),
            "99-й перцентиль": round(percentile_values[0.99], 2),
            "Распределение": range_info,
            "Вывод": distribution
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df
    
    
    
    