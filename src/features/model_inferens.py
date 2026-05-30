import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import mlflow.pyfunc
from mlflow.models import infer_signature
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictModel:
    def __init__(self):
        print("=" * 50)
        print("ИНИЦИАЛИЗАЦИЯ PredictModel")
        print("=" * 50)
        
        # Определяем корень проекта
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        
        
        
        # Вариант 3: Загрузка из файла .pkl
       
        try:
            model_file = project_root / "models" / "model_v1.pkl"
            print(f"\n🔄 Попытка загрузить из файла: {model_file}")
            print(f"📁 Файл существует? {model_file.exists()}")
            
            if model_file.exists():
                self.model = joblib.load(model_file)
                if self.model is not None:
                    print(f"✅ Модель загружена из {model_file}")
            else:
                print(f"❌ Файл не найден")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки из файла: {e}")
        
        if self.model is None:
            print("\nКРИТИЧЕСКАЯ ОШИБКА: МОДЕЛЬ НЕ ЗАГРУЖЕНА!")
            print("Проверьте:")
            print("1. Название модели в MLflow Registry")
            print("2. Наличие файла модели в папке models/")
            print("3. Путь к mlruns")
            raise Exception("Модель не загружена")
        
        print(f"\nТип модели: {type(self.model)}")
        print("=" * 50)
                
        
    def engineer_features(self, df):
        """Создание новых признаков (feature engineering)"""
        data = df.copy()
        
        # Показывает «лояльность» сотрудника к компании
        data['TenureRatio'] = np.where(
            data['TotalWorkingYears'] > 0,
            data['YearsAtCompany'] / data['TotalWorkingYears'],
            0
        )
        
        # Комбинирует статус и вовлечённость
        data['RoleEngagementScore'] = data['JobLevel'] * data['JobInvolvement']
        
        # Измеряет «специализацию» или «застой»
        data['RoleStabilityRatio'] = np.where(
            data['TotalWorkingYears'] > 0,
            data['YearsInCurrentRole'] / data['TotalWorkingYears'], 
            0
        )
        
        # Индикатор стабильности отношений
        data['ManagerStabilityRatio'] = np.where(
            data['YearsInCurrentRole'] > 0,
            data['YearsWithCurrManager'] / data['YearsInCurrentRole'],
            1  
        )
        
        # Как быстро сотрудник растёт в компании
        data['PromotionRate'] = np.where(
            data['YearsAtCompany'] > 0,
            data['JobLevel'] / data['YearsAtCompany'],
            0
        )
        
        # Доход на единицу опыта
        data['IncomePerYear'] = np.where(
            data['TotalWorkingYears'] > 0,
            data['MonthlyIncome'] / data['TotalWorkingYears'],
            0
        )
        
        # Ожидания vs реальность
        data['SalaryHikeExpectation'] = data['PercentSalaryHike'] - data['PerformanceRating']
        
        return data
    
    def create_preprocessor(self, data):
        """Создание препроцессора (как при обучении)"""
        # Определяем колонки по типам
        features = data.columns.tolist()
        normalize_features = [i for i in features if i not in ['Gender', 'OverTime', 'BusinessTravel', 'Over18',
                                                               'Department', 'EducationField', 'JobRole', 'MaritalStatus']]
        
        binary_columns = ['Gender', 'OverTime']
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        
        preprocessor = ColumnTransformer([
            # Числовые признаки: заполняем пропуски медианой → нормализуем
            ('normalize', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), normalize_features),
            
            # Бинарные признаки: заполняем пропуски модой → one-hot кодируем
            ('binary', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', dtype=int, handle_unknown='ignore'))
            ]), binary_columns),
            
            # Категориальные признаки: заполняем пропуски модой → ordinal кодируем
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_columns)
        ])
        
        return preprocessor
        
    def preprocess(self, df):
        """Полная предобработка данных (как при обучении)"""
        # Шаг 1: Создаем новые признаки
        df = self.engineer_features(df)
        print(f"📊 После engineer_features: {df.shape[1]} колонок")
        
        # Шаг 2: Удаляем ненужные колонки (КАК ПРИ ОБУЧЕНИИ!)
        columns_to_drop = ['MonthlyIncome', 'TotalWorkingYears', 'YearsInCurrentRole', 
                          'YearsWithCurrManager', 'PercentSalaryHike']
        
        existing_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_to_drop, axis=1)
        print(f"🗑️ Удалены колонки: {existing_to_drop}")
        print(f"📊 После удаления: {df.shape[1]} колонок")
        
        # Шаг 3: Создаем препроцессор
        preprocessor = self.create_preprocessor(df)
        
        # Шаг 4: Применяем препроцессор
        X_processed = preprocessor.fit_transform(df)
        print(f"📊 После трансформации: {X_processed.shape[1]} признаков")
        
        return X_processed
        
    def predict(self, input_data):
        X_processed = self.preprocess(input_data)
        
        # Делаем предсказание
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return {
            'prediction': int(predictions[0]),  # 0 или 1
            'probability': float(probabilities[0]),  # вероятность ухода
            'will_leave': bool(predictions[0] == 1)
        }

def main():
    model = '../../models/model_v1.pkl'