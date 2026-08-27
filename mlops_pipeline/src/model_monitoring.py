# Librerias base para manejo funciones/pipelines
import numpy as np
import pandas as pd
import os

# Para calculos de métricas - Validación de Data Drift
from scipy.stats import ks_2samp, chi2_contingency

# Librerias de generador de imagenes/graficas
import matplotlib
matplotlib.use('Agg')  # Genera la imagen sin intentar abrir ventana Tkinter
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn para modelado/limpieza/procesamiento de datos, para entrenar modelos ML
from sklearn.model_selection import train_test_split

# Metodos Generados en pipelines anteriores (cargar_datos)
try:
    from src.cargar_datos import cargarDatos
except ModuleNotFoundError:
    from cargar_datos import cargarDatos

# Definición de métricas para Data Drift
def calcular_psi(reference, current, num_bins=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones continuas.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(reference, quantiles)
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    bins[0] -= 1e-5
    bins[-1] += 1e-5

    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / len(reference)
    curr_pct = curr_counts / len(current)

    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_val)

# Métricas para variables categoricas
def calcular_chi_cuadrado(reference, current):
    """
    Calcula la prueba Chi-cuadrado para variables categóricas garantizando el tipo str.
    """
    ref_str = pd.Series(reference).astype(str)
    curr_str = pd.Series(current).astype(str)

    ref_counts = ref_str.value_counts()
    curr_counts = curr_str.value_counts()

    tabla = pd.concat([ref_counts, curr_counts], axis=1, keys=['ref', 'curr']).fillna(0)

    if tabla.empty or len(tabla) < 2:
        return 1.0

    chi2, p_val, _, _ = chi2_contingency(tabla.T)
    return float(p_val)

# Evaluación final. Simulación con dataset original comparando entrenamiento vs test
def evaluar_drift_dataset(df_ref, df_new, alpha=0.05):
    """
    Analiza y compara df_ref y df_new para detectar Data Drift por columna.
    """
    resultados = []

    for col in df_ref.columns:
        if col not in df_new.columns:
            continue

        ref_series = df_ref[col].dropna()
        curr_series = df_new[col].dropna()

        if len(ref_series) == 0 or len(curr_series) == 0:
            continue

        # Validación nativa de Pandas para tipos numéricos
        if pd.api.types.is_numeric_dtype(df_ref[col]):
            ks_stat, p_val = ks_2samp(ref_series, curr_series)
            psi_val = calcular_psi(ref_series, curr_series)
            metrica_principal = f"KS: {ks_stat:.4f}"

            if psi_val >= 0.2 or p_val < alpha:
                estado = "CRÍTICO"
            elif psi_val >= 0.1:
                estado = "ADVERTENCIA"
            else:
                estado = "ESTABLE"

            resultados.append({
                'Variable': col,
                'Tipo': 'Numérica',
                'Métrica': metrica_principal,
                'P_Value': round(float(p_val), 4),
                'PSI': round(psi_val, 4),
                'Estado': estado
            })

        # Si la columna es categórica/objeto/string
        else:
            p_val = calcular_chi_cuadrado(ref_series, curr_series)

            if p_val < alpha:
                estado = "CRÍTICO"
            elif p_val < alpha * 2:
                estado = "ADVERTENCIA"
            else:
                estado = "ESTABLE"

            resultados.append({
                'Variable': col,
                'Tipo': 'Categórica',
                'Métrica': 'Chi-Cuadrado',
                'P_Value': round(float(p_val), 4),
                'PSI': 'N/A',
                'Estado': estado
            })

    return pd.DataFrame(resultados)

# Se nos permitio acceder a un nuevo DataSet simulado para comprobar si hay Data Drift
# Se reemplazaran algunas lineas de codigo con este nuevo DataSet para revisar si las métricas previamente generan
# Alertas o si el sistema aun visualiza que todos los datos estan estables.

if __name__ == "__main__":
    print("🕵️ VERIFICACIÓN DE DATA DRIFT")

    # DataSet original
    df_ref = cargarDatos()

    # DataSet simulado para Data Drift
    df_new = pd.read_excel('Base_de_datos_con_Data_Drift_Simulado.xlsx')

    # Evualamos si hay Data Drift
    reporte = evaluar_drift_dataset(df_ref, df_new)

    print("\n--------------------- REPORTE DE DATA DRIFT ---------------------")
    print(reporte)
    print("✅ Validación de existencia de Data Drift realizado con exito!")