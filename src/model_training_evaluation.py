import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score, 
    confusion_matrix, precision_recall_curve
)

# Algoritmos a evaluar
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# Importar el pipeline de Feature Engineering que construiste previamente
from src.ft_engineering import (
    cargarDatos, construir_pipeline_preprocesamiento, 
    num_cols, cat_nom_cols, cat_ord_cols, orden_ordinales
)


# ==========================================
# 1. FUNCIONES REQUERIDAS POR EL CURSO
# ==========================================

def build_model(model_name, **kwargs):
    """
    Construye e instancia un modelo de clasificación configurado con class_weight='balanced'
    para tratar el desbalanceo (5% / 95%).
    """
    modelos_disponibles = {
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42, **kwargs),
        'ExtraTrees': ExtraTreesClassifier(class_weight='balanced', random_state=42, **kwargs),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, **kwargs),
        'GradientBoosting': GradientBoostingClassifier(random_state=42, **kwargs)
    }
    
    if model_name not in modelos_disponibles:
        raise ValueError(f"Modelo {model_name} no reconocido. Elige entre {list(modelos_disponibles.keys())}")
        
    return modelos_disponibles[model_name]


def summarize_classification(y_true, y_pred, y_probs, model_name="Modelo"):
    """
    Calcula y retorna las métricas clave de evaluación para clasificación desbalanceada.
    """
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    metrics_summary = {
        'Modelo': model_name,
        'F1-Score': round(f1, 4),
        'Recall': round(recall, 4),
        'Precision': round(precision, 4),
        'ROC-AUC': round(roc_auc, 4)
    }
    
    return metrics_summary


# ==========================================
# 2. AJUSTE DE UMBRAL DE DECISIÓN
# ==========================================

def optimizar_umbral_decision(y_true, y_probs):
    """
    Encuentra el umbral óptimo de probabilidad que maximiza el F1-Score en lugar del corte por defecto (0.5).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.where((precisions + recalls) > 0, 2 * (precisions * recalls) / (precisions + recalls), 0)
    
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    
    return best_threshold


# ==========================================
# 3. VISUALIZACIONES COMPARATIVAS
# ==========================================

def graficar_comparativa(df_resumen):
    """
    Genera gráficos comparativos de barras para visualizar el desempeño de los modelos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: F1-Score y Recall
    df_melt = df_resumen.melt(id_vars=['Modelo'], value_vars=['F1-Score', 'Recall'], 
                              var_name='Métrica', value_name='Valor')
    sns.barplot(data=df_melt, x='Modelo', y='Valor', hue='Métrica', ax=axes[0], palette='viridis')
    axes[0].set_title('Comparativa de F1-Score y Recall (Con Umbral Ajustado)')
    axes[0].set_ylim(0, 1)
    
    # Gráfico 2: ROC-AUC
    sns.barplot(data=df_resumen, x='Modelo', y='ROC-AUC', ax=axes[1], palette='magma')
    axes[1].set_title('Comparativa de ROC-AUC')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ==========================================

def ejecutar_pipeline_entrenador():
    # 1. Cargar datos
    df = cargarDatos()
    X = df.drop(columns=['Pago_atiempo', 'fecha_prestamo'])
    y = df['Pago_atiempo']
    
    # 2. Split Stratified (mantiene el 5%/95% en train y test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Aplicar Feature Engineering
    pipeline_prep = construir_pipeline_preprocesamiento(num_cols, cat_nom_cols, cat_ord_cols, orden_ordinales)
    X_train_proc = pipeline_prep.fit_transform(X_train)
    X_test_proc = pipeline_prep.transform(X_test)
    
    # 4. Lista de modelos a entrenar
    nombres_modelos = ['RandomForest', 'ExtraTrees', 'LogisticRegression', 'GradientBoosting']
    resumen_resultados = []
    modelos_entrenados = {}
    
    for name in nombres_modelos:
        # Construir
        model = build_model(name)
        
        # Entrenar (Manejo especial de sample_weight para GradientBoosting)
        if name == 'GradientBoosting':
            class_weights = len(y_train) / (2.0 * np.bincount(y_train))
            weights = np.array([class_weights[int(i)] for i in y_train])
            model.fit(X_train_proc, y_train, sample_weight=weights)
        else:
            model.fit(X_train_proc, y_train)
            
        # Probabilidades predichas
        y_probs = model.predict_proba(X_test_proc)[:, 1]
        
        # Ajuste de Umbral
        umbral_optimo = optimizar_umbral_decision(y_test, y_probs)
        y_pred_opt = (y_probs >= umbral_optimo).astype(int)
        
        # Evaluar
        metrics = summarize_classification(y_test, y_pred_opt, y_probs, model_name=name)
        metrics['Umbral_Optimo'] = round(umbral_optimo, 4)
        
        resumen_resultados.append(metrics)
        modelos_entrenados[name] = model
        
    # 5. Tabla resumen
    df_resumen = pd.DataFrame(resumen_resultados).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    print("\n================ TABLA RESUMEN DE EVALUACIÓN ================")
    print(df_resumen)
    
    # 6. Gráficos
    graficar_comparativa(df_resumen)
    
    return df_resumen, modelos_entrenados


if __name__ == "__main__":
    ejecutar_pipeline_entrenador()