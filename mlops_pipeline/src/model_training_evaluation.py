# Librerias base para manejo funciones/pipelines
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Genera la imagen sin intentar abrir ventana Tkinter
import matplotlib.pyplot as plt
import seaborn as sns

# Librerias para entrenado y parametros Modelo ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score, 
    precision_recall_curve
)

# Algoritmos a evaluar (Tipo modelos ML)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler # Para realizar Undersampler

# Importar el pipeline de Feature Engineering
try:
    from src.cargar_datos import cargarDatos
    from src.ft_engineering import construir_pipeline_preprocesamiento
except ModuleNotFoundError:
    from cargar_datos import cargarDatos
    from ft_engineering import construir_pipeline_preprocesamiento

# 1. Construyendo el modelo
def build_model(model_name, **kwargs):

    modelos_disponibles = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight=None, random_state=42, n_jobs=2, **kwargs
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, max_depth=10, class_weight=None, random_state=42, n_jobs=2, **kwargs
        ),
        'LogisticRegression': LogisticRegression(
            class_weight=None, max_iter=1000, random_state=42, n_jobs=2, **kwargs
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42, **kwargs
        )
    }
    
    if model_name not in modelos_disponibles:
        raise ValueError(f"Modelo {model_name} no reconocido.")
        
    return modelos_disponibles[model_name]

# 2. Optimización de umbral para tomar en cuenta el mejor modelo
def optimizar_umbral_decision(y_true, y_probs):
    y_probs_moroso = 1 - y_probs
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs_moroso, pos_label=0)
    f1_scores = np.where((precisions + recalls) > 0, 2 * (precisions * recalls) / (precisions + recalls), 0)
    best_idx = np.argmax(f1_scores[:-1])

    # Retornando el umbral equivalente para la probabilidad de Buen Pagador (clase 1)
    return 1 - thresholds[best_idx]

# 3. Evaluación de los modelos umbral defaul & umbral optimizado
# Tomando en cuenta Escalabilidad, Consistencia y Performance
def evaluate_model_full(model, X_train, y_train, X_test, y_test, model_name="Modelo"):

    # Escalabilidad: entrenamiento y tiempos
    start_train = time.time()
    if model_name == 'GradientBoosting':
        class_weights = len(y_train) / (2.0 * np.bincount(y_train))
        weights = np.array([class_weights[int(i)] for i in y_train])
        model.fit(X_train, y_train, sample_weight=weights)
    else: 
        model.fit(X_train, y_train)
    train_time_sec = round(time.time() - start_train, 4)

    # Tiempo de Inferencia por muestra
    start_infer = time.time()
    y_probs_test = model.predict_proba(X_test)[:, 1]
    infer_time_ms = round(((time.time() - start_infer) / len(X_test)) * 1000, 4)

    # 1. Probabilidades de train para calcular el umbral
    y_probs_train = model.predict_proba(X_train)[:, 1]
    umbral_optimo = optimizar_umbral_decision(y_train, y_probs_train)

    # 2. Evaluación en TEST con el umbral precalculado
    y_pred_def = (y_probs_test >= 0.5).astype(int)
    f1_def = round(f1_score(y_test, y_pred_def), 4)
    recall_def = round(recall_score(y_test, y_pred_def), 4)

    y_pred_opt = (y_probs_test >= umbral_optimo).astype(int)
    f1_opt = round(f1_score(y_test, y_pred_opt), 4)
    recall_opt = round(recall_score(y_test, y_pred_opt), 4)
    precision_opt = round(precision_score(y_test, y_pred_opt), 4)
    roc_auc = round(roc_auc_score(y_test, y_probs_test), 4)

    # Consistencia: Brecha de Sobreajuste (Overfit Gap)
    y_pred_train = (y_probs_train >= umbral_optimo).astype(int)
    f1_train = f1_score(y_train, y_pred_train)
    overfit_gap = round(f1_train - f1_opt, 4)

    # Métricas clase minoritaria (0: Morosos)
    f1_morosos_umbral_opt = round(f1_score(y_test, y_pred_opt, pos_label=0), 4)
    recall_morosos_umbral_opt = round(recall_score(y_test, y_pred_opt, pos_label=0), 4)

    return {
        'Modelo': model_name,

        # Comparativa de Umbrales
        'F1_Def(0.5)': f1_def,
        'F1_Opt': f1_opt,
        'Recall_Def': recall_def,
        'Recall_Opt': recall_opt,
        'Umbral_Opt': round(umbral_optimo, 4),

        # General Performance
        'Precision_Opt': precision_opt,
        'ROC-AUC': roc_auc,

        # Evaluación de Morosos (0 - 5%)
        'F1_Morosos': f1_morosos_umbral_opt,
        'Recall_Morosos': recall_morosos_umbral_opt,

        # Consistency
        'Overfit_Gap': overfit_gap,

        # Scalability
        'Train_Sec': train_time_sec,
        'Infer_ms': infer_time_ms
    }

# 4. Grafica Comparativa
def graficar_comparativa(df_resumen):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico 1 Impacto del Umbral en F1-Score
    df_f1 = df_resumen.melt(id_vars=['Modelo'], value_vars=['F1_Def(0.5)', 'F1_Opt'], 
                            var_name='Umbral', value_name='F1-Score')
    sns.barplot(data=df_f1, x='Modelo', y='F1-Score', hue='Umbral', ax=axes[0], palette='PuBu')
    axes[0].set_title('Impacto del Umbral de Decisión en F1-Score')
    axes[0].set_ylim(0, 1)
    
    # Gráfico 2: ROC-AUC por Modelo
    sns.barplot(data=df_resumen, x='Modelo', y='ROC-AUC', ax=axes[1], palette='viridis')
    axes[1].set_title('Capacidad de Discriminación (ROC-AUC)')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('comparativa_modelos.png')
    print("\n[OK] Gráfica guardada exitosamente como 'comparativa_modelos.png'")


# Observaciones importantes generadas en el entrenamiento anterior de los modelos.
# Evidencias: Se visualiza un F1 por encima del 95% y un ROC-AUC por encima del 95%, dando a entender
# Que los modelos estan funcionando "muy bien" a pesar de contar con un desbalance 95% clientes que si pagaron
# 5% clientes morosos. Se evidencias 2 posibilidades: 
# -----1: El modelo esta acertando la clase facil o la de mayor cantidad (95%) a pesar de realizar un balanceo en los parametros
# -----2: Es posible que alguna variable en X, este altamente correlacionada con la variable objetivo y que éste este 
# generando Data Leakage


# ------------------------------ Validación del DataSet en el EDA --------------------------------
# Se revisa el EDa y se visualiza la columna Puntaje, la cual tiene una alta correlación con la variable objetivo
# Esta columna es un número canculado a los clientes despues de realizar un pago, es decir, es una variable calculada
# POST - EVENTO (despues de la variable objetivo), generando Data Leakage y un F1 "artificial".

# ------------------------------ Columnas Correlacionadas entre Sí -------------------------------
# - Se conservan desagregados 'creditos_sectorFinanciero/Real/Cooperativo'.

# ------------------------------ Columnas eliminadas ---------------------------------------------
# 1- Se elimina 'fecha_prestamo' (no relevante para modelos tabulares sin serie temporal).
# 2. Se elimina 'puntaje' (Fuga de datos (Data Leakage post-evento), correlación de = 0.79).
# 3. Se elimina 'cant_creditosvigentes' (Redundante, es la suma de creditos_sectorFinanciero, Real y Cooperativo).
# 4. Se elimina 'saldo_principal' (Redundante con 'saldo_total')

# 5. Ejecución Principal del Pipeline de Entrenamiento
def ejecutar_pipeline_entrenador():
    df = cargarDatos()
    df_copy = df.copy()

    # Selección explícita de columnas a remover para evitar Data Leakage
    COLS_A_ELIMINAR = [
        'puntaje', 
        'fecha_prestamo', 
        'cant_creditosvigentes', 
        'saldo_principal'
    ]

    X = df_copy.drop(columns=['Pago_atiempo'] + COLS_A_ELIMINAR, errors='ignore')
    y = df_copy['Pago_atiempo']

    # Divisón Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Aplicación del Pipeline de Preprocesamiento Dinámico
    pipeline_prep = construir_pipeline_preprocesamiento()
    X_train_proc = pipeline_prep.fit_transform(X_train)
    X_test_proc = pipeline_prep.transform(X_test)

    # Confirmación visual de las dimensiones reales en consola
    print(f"\n[INFO] Matriz procesada de Entrenamiento (31 cols esperadas): {X_train_proc.shape}")
    print(f"[INFO] Matriz procesada de Prueba: {X_test_proc.shape}")

    # !!! Aplicando Undersampling ÚNICAMENTE a los datos de entrenamiento (X_train_proc, y_train)
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42) 

    # sampling_strategy=0.5 deja una relación de 1 moroso por cada 2 pagadores en train
    X_train_res, y_train_res = rus.fit_resample(X_train_proc, y_train)
    
    nombres_modelos = ['RandomForest', 'ExtraTrees', 'LogisticRegression', 'GradientBoosting']
    resumen_resultados = []
    modelos_entrenados = {}
    
    for name in nombres_modelos:
        model = build_model(name)
        metrics = evaluate_model_full(model, X_train_res, y_train_res, X_test_proc, y_test, model_name=name)
        
        resumen_resultados.append(metrics)
        modelos_entrenados[name] = model
        
    df_resumen = pd.DataFrame(resumen_resultados).sort_values(by='F1_Opt', ascending=False).reset_index(drop=True)
    
    print("\n--------------------- EVALUACIÓN MODELOS (UMBRAL): DEFAULT vs OPTIMIZADO ---------------------")
    print(df_resumen.to_string())
    
    graficar_comparativa(df_resumen)

    # -------------------------- SELECCIÓN DEL MEJOR MODELO --------------------------

    # Para importar/extraer el mejor modelo
    import joblib
    
    # Exportar el modelo ganador seleccionado (ExtraTrees)
    modelo_ganador = modelos_entrenados['ExtraTrees']

    # Guardando tanto el modelo como el pipeline de preprocesamiento
    joblib.dump(modelo_ganador, 'src/modelo_ganador.pkl')
    joblib.dump(pipeline_prep, 'src/construir_pipeline_preprocesamiento.pkl')
    print("\n[ÉXITO] Modelo ExtraTrees y Pipeline exportados correctamente en 'src/'")
    
    return df_resumen, modelos_entrenados


if __name__ == "__main__":
    ejecutar_pipeline_entrenador()

# Conclusiones:
# Al aplicar un submuestreo estratégico (Undersampling 2:1) en el conjunto de entrenamiento junto con los ratios 
# financieros calculados y la optimización de umbrales en la clase minoritaria
# Pasamos de detectar solo un 7.8% de los morosos a un 62.7% en Regresión Logística y un 58.8% en Random Forest.
# Logrando un modelo genuinamente útil para la mitigación del riesgo en producción.
# 🏆 Modelo Ganador según métricas y profundidad: ExtraTrees.