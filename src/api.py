from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import os
import io

app = FastAPI(
    title="Scoring API",
    description="Платформа для массового скоринга клиентов телеком-компании",
)

# Загрузка модели при старте
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.4059 

@app.post("/score_clients", summary="Массовый скоринг клиентов (Загрузка CSV)")
async def score_clients(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"message": "Загрузите файл формата .csv"})
    
    # Чтение файла
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    if 'ID' not in df.columns:
        return JSONResponse(status_code=400, content={"message": "В файле обязательно должна быть колонка 'ID'"})
    
    # Отделяем признаки от ID
    model_features = model.feature_names_in_
    X = df[model_features]
    
    # Предикт
    probas = model.predict_proba(X)[:, 1]
    
    # Формирование детального отчета
    detailed_report = []
    for i, proba in enumerate(probas):
        will_leave = bool(proba >= THRESHOLD)
        detailed_report.append({
            "client_id": int(df.iloc[i]['ID']),
            "вердикт_модели": "В ЗОНЕ РИСКА (Уйдет)" if will_leave else "ЛОЯЛЕН (Останется)",
            "вероятность_оттока": f"{proba * 100:.1f}%",
            "рекомендация": "Срочно предложить персональную скидку!" if will_leave 
            else "Дополнительных действий не требуется."
        })
            
    return {
        "обработано_клиентов": len(df),
        "отчет_по_клиентам": detailed_report
    }