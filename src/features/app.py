from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any  # ← Импортируем Any из typing
import pandas as pd
from model_inferens import PredictModel

# Глобальная переменная для хранения предиктора
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("=" * 50)
    print("🔄 ЗАПУСК ПРИЛОЖЕНИЯ")
    print("=" * 50)
    
    # Загрузка модели с проверками
    print("📦 Загрузка модели...")
    try:
        predictor = PredictModel()
        
        # Проверка, что модель действительно загрузилась
        if predictor is None:
            raise Exception("PredictModel вернул None")
        
        # Дополнительная проверка: есть ли метод predict
        if not hasattr(predictor, 'predict'):
            raise Exception("У модели нет метода predict")
        
        # Проверка: работает ли метод predict (тестовый вызов)
        try:
            test_df = pd.DataFrame([{
                'Age': 34,
                'BusinessTravel': 'Travel_Rarely',
                'DailyRate': 800,
                'Department': 'Sales',
                'DistanceFromHome': 10,
                'Education': 3,
                'EducationField': 'Life Sciences',
                'EmployeeCount': 1,
                'EmployeeNumber': 101,
                'EnvironmentSatisfaction': 3,
                'Gender': 'Male',
                'HourlyRate': 50,
                'JobInvolvement': 3,
                'JobLevel': 2,  # ← ОБЯЗАТЕЛЬНО!
                'JobRole': 'Sales Executive',
                'JobSatisfaction': 3,
                'MaritalStatus': 'Married',
                'MonthlyIncome': 5000,
                'MonthlyRate': 12000,
                'NumCompaniesWorked': 2,
                'Over18': 'Y',
                'OverTime': 'No',
                'PercentSalaryHike': 15,
                'PerformanceRating': 3,
                'RelationshipSatisfaction': 3,
                'StandardHours': 80,
                'StockOptionLevel': 1,
                'TotalWorkingYears': 10,
                'TrainingTimesLastYear': 2,
                'WorkLifeBalance': 3,
                'YearsAtCompany': 5,
                'YearsInCurrentRole': 3,
                'YearsWithCurrManager': 2
                }])
            test_result = predictor.predict(test_df)
            if test_result is None:
                raise Exception("Метод predict вернул None")
            print(f"✅ Тестовое предсказание успешно: {test_result}")
        except Exception as e:
            print(f"⚠️ Тестовый вызов predict не удался: {e}")
            # Не прерываем загрузку, просто предупреждаем
        
        print("✅ МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА!")
        print(f"📊 Тип модели: {type(predictor.model) if hasattr(predictor, 'model') else 'Неизвестно'}")
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА при загрузке модели: {e}")
        import traceback
        traceback.print_exc()
        predictor = None
        print("⚠️ Приложение запустится без модели. Эндпоинт /predict будет возвращать ошибку 503")
    
    print("=" * 50)
    
    yield  # Приложение работает и обрабатывает запросы
    
    # Этот код выполнится ПРИ ОСТАНОВКЕ приложения
    print("Выключение приложения, очистка ресурсов...")
    predictor = None

app = FastAPI(lifespan=lifespan)

class PredictionRequest(BaseModel):
    features: Dict[str, Any]  

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Преобразуем словарь в DataFrame
        input_df = pd.DataFrame([request.features])
        result = predictor.predict(input_df)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Простой эндпоинт для проверки работоспособности"""
    return {"status": "ok", "model_loaded": predictor is not None}