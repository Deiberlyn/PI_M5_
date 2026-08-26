# Librerias base para manejo de los DataSet
import pandas as pd

# Libreria para levantamiento del modelo y APIs
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# 1. ------ Iniciamos FastAPI
app = FastAPI(
    title="API de Evaluación de Riesgo Crediticio",
    description= "Endpoint para la predicción de scoring crediticio usando ExtraTrees",
    version="1.0.0"
)

# 2. ------ Cargar artifacts (Modelo y Pipeline previamente guardados en .pkl)
try:
    modelo = joblib.load("src/modelo_ganador.pkl")
    pipeline = joblib.load("src/construir_pipeline_preprocesamiento.pkl")

except Exception:
    # Ajuste por si se ejecuta desde dentro del contenedor Docker, o no entiende la ruta
    modelo = joblib.load("modelo_ganador.pkl")
    pipeline = joblib.load("construir_pipeline_preprocesamiento.pkl")

@app.get("/")
def home():
    return {"mensaje": "API de Predicción de Riesgo Crediticio Activa 🚀"}

@app.post("/predict")
def predecir(datos: List[Dict[str, Any]]):
    """
    Endpoint para realizar predicciones individuales o por lote (batch).
    Recibe una lista de diccionarios JSON con las características del cliente.
    """
    try:

        # Convirtiendo datos recibidos en DataFrame
        df_input = pd.DataFrame(datos)

        # Selección explícita de columnas a remover para evitar Data Leakage
        COLS_A_ELIMINAR = [
            'puntaje', 
            'fecha_prestamo', 
            'cant_creditosvigentes', 
            'saldo_principal',
            'Pago_atiempo'
        ]

        df_clean = df_input.drop(columns=COLS_A_ELIMINAR, errors='ignore')

        # Aplicando pipeline de preprocesamiento
        X_proc = pipeline.transform(df_clean)

        # Prediciendo probabilidades y clase
        probs = modelo.predict_proba(X_proc)[:,1]

        # Umbral ajustado segun el entrenamiento realizado en model_training_evaluation.py
        umbrales = (probs >= 0.6553).astype(int)

        resultados = []

        for i, (prob, pred) in enumerate(zip(probs, umbrales)):
            resultados.append({
                "registro_id": i + 1,
                "probabilidad_buen_pagador": round(float(prob), 4),
                "prediccion": int(pred),
                "estado_credito": "Aprobado (Buen Pagador)" if pred == 1 else "Rechazado (Riesgo Moroso)"
            })
        return {"predicciones": resultados}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el procesamiento: {str(e)}")
