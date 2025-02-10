import pandas as pd

def load_data_csv(file_path):
    try:
        csv_file = pd.read_csv(file_path)
        print("Данные успешно загружены.")
        return csv_file
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)

        
def load_data_json(file_path):
    try:
        json_file = pd.read_json(file_path)
        print("Данные успешно загружены.")
        return json_file
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1) 

    

