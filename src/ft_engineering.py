# Librerias para el tratamiento del DataSet
import pandas as pd
import numpy as np

# Librerias y nucleos para la estrutura de limpieza automatica en el pipeline para Modelo ML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Importar la función de cargarDatos donde ya tenemos el DataSet
from src.cargar_datos import cargarDatos

# 1. Empezando con el pipeline (LIMPIEZA Y REGLAS DE NEGOCIO) siguiendo los pasos realizados en el EDA
class LimpiadorPersonalizado(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit (self, X, y=None):
        return self

    def transform (self, X):
        X_out = X.copy()

        # 1. EDAD: Inconsistencias de 'edad_cliente' (datos presentan edades mayores a 100 años), se transforma a NaN
        if 'edad_cliente' in X_out.columns:
            X_out.loc[X_out['edad_cliente'] > 90, 'edad_cliente'] = np.nan

        # 2. PUNTAJE y PUNTAJE_DATACREDITO: Valores negativos a NaN
        for col in ['puntaje', 'puntaje_datacredito']:
            if col in X_out.columns:
                X_out.loc[X_out[col] < 0, col] = np.nan

        # 3. TENDENCIA_INGRESOS: Limpieza de categorías no válidas y de valores numericos o incongruentes
        if 'tendencia_ingresos' in X_out.columns:
            categorias_validas = ['Creciente', 'Decreciente', 'Estable']
            X_out['tendencia_ingresos'] = X_out['tendencia_ingresos'].apply(
                lambda x: x if x in categorias_validas else 'Sin_Informacion'
            )

        # 4 Blindando el pipeline a posibles errores del futuro normalizando formato
        if 'tipo_laboral' in X_out.columns:
            categorias_laborales_validas = ['Empleado', 'Independiente']
            X_out['tipo_laboral'] = X_out['tipo_laboral'].apply(
                lambda x: x if x in categorias_laborales_validas else 'Sin_Informacion'
            )

        # 5. IMPUTACIÓN AGRUPADA POR TIPO_LABORAL (Mediana)
        if 'tipo_laboral' in X_out.columns:
            cols_agrupadas = ['edad_cliente', 'saldo_total', 'saldo_principal', 'promedio_ingresos_datacredito']
            for col in cols_agrupadas:
                if col in X_out.columns:
                    X_out[col] = X_out.groupby('tipo_laboral')[col].transform(
                        lambda x: x.fillna(x.median())
                    )

        return X_out

# 2. Construcción del Pipeline Principal
def construir_pipeline_preprocesamiento(num_cols, cat_nom_cols, cat_ord_cols, orden_ordinales):

    # Pipeline para variables numéricas globales (saldos en mora y medianas generales)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas nominales (tipo_laboral)
    cat_nom_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Sin_Informacion')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline para variables categóricas ordinales (tendencia_ingresos)
    cat_ord_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Sin_Informacion')),
        ('encoder', OrdinalEncoder(categories=orden_ordinales, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Ensamble con ColumnTransformer
    preprocesador = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat_nom', cat_nom_pipeline, cat_nom_cols),
        ('cat_ord', cat_ord_pipeline, cat_ord_cols)
    ])

    # Pipeline integrador final
    pipeline_completo = Pipeline(steps=[
        ('limpieza_inicial', LimpiadorPersonalizado()),
        ('preprocesamiento', preprocesador)
    ])

    return pipeline_completo

# 3. Prueba de Ejecución
if __name__ == "__main__":
    df = cargarDatos()

    # Generando una copia del df para realizar las divisiones correspondientes sin afectar los datos originales
    df_copy = df.copy()

    # Separar la variable objetivo (y) y campos no requeridos de las características (X)
    X = df_copy.drop(columns=['Pago_atiempo', 'fecha_prestamo'])
    y = df_copy['Pago_atiempo']

    # 1. Variables Numéricas Continuas
    num_cols = [
        'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
        'total_otros_prestamos', 'cuota_pactada', 'puntaje', 'puntaje_datacredito',
        'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total',
        'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero',
        'creditos_sectorCooperativo', 'creditos_sectorReal', 'promedio_ingresos_datacredito'
    ]

    # 2. Categóricas Nominales -> Tratadas con One-Hot Encoding
    cat_nom_cols = ['tipo_laboral', 'tipo_credito']

    # 3. Categóricas Ordinales -> Tratadas con OrdinalEncoder
    cat_ord_cols = ['tendencia_ingresos']

    # Definición explícita de jerarquía para la variable ordinal
    orden_tendencia = ['Decreciente', 'Sin_Informacion', 'Creciente', 'Estable']
    orden_ordinales = [orden_tendencia]

    # Instanciación y ejecución del Pipeline
    pipeline = construir_pipeline_preprocesamiento(num_cols, cat_nom_cols, cat_ord_cols, orden_ordinales)
    
    X_procesado = pipeline.fit_transform(X)
    print("¡Pipeline ejecutado exitosamente!")
    print("Forma final de la matriz procesada (X):", X_procesado.shape)
    print("Forma de la variable objetivo (y):", y.shape)

    # ---- Configurando estado de los datos procesados en la consola ----
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Reconstruyendo los nombres de las columnas para la inspección de los datos procesados
    nombres_cat_nom = list(
        pipeline.named_steps['preprocesamiento']
        .named_transformers_['cat_nom']
        .named_steps['encoder']
        .get_feature_names_out(cat_nom_cols)
    )

    nombres_cols = num_cols + nombres_cat_nom + cat_ord_cols
    df_resultado = pd.DataFrame(X_procesado, columns = nombres_cols)

    # Verificación de la codificación ordinal de tendencia_ingresos
    print("--- CONTEO Y VALORES EN TENDENCIA_INGRESOS (0: Dec, 1: Sin_Info, 2: Crec, 3: Est) ---")
    print(df_resultado['tendencia_ingresos'].value_counts().sort_index())

    print("\n--- PRIMERAS 5 FILAS DEL DATAFRAME TRANSFORMADO ---")
    print(df_resultado.head())