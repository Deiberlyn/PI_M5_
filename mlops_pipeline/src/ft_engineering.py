# Librerias para el tratamiento del DataSet
import pandas as pd
import numpy as np

# Librerias y nucleos para la estrutura de limpieza automatica en el pipeline para Modelo ML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

# Importar la función de cargarDatos donde ya tenemos el DataSet
from src.cargar_datos import cargarDatos

# Se realizan cambios en el pipeline de preprocesamiento.
# Ya que anteriormente tenia errores en el script, estando "hardcodeado", lo que impedia 
# El entrenamiento de los modelos cambiando/eliminando variables de X que estaban "hardcodeados"
# Tomando en cuenta ésto, tambien se agregan a la prueba de ejecución lineas de codigo de simulación
# Tales como separar variables, separar target. Probar Instancias y probar la transformación

# CONSTANTES GLOBALES DE REFERENCIA Y REGLAS ORDINALES
ORDEN_TENDENCIA = ['Decreciente', 'Sin_Informacion', 'Creciente', 'Estable']
ORDEN_ORDINALES = [ORDEN_TENDENCIA]

# 1. Empezando con el pipeline (LIMPIEZA Y REGLAS DE NEGOCIO) siguiendo los pasos realizados en el EDA
class LimpiadorPersonalizado(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. EDAD: Inconsistencias (mayores a 90 años se transforman a NaN para ser imputadas)
        if 'edad_cliente' in X_out.columns:
            X_out.loc[X_out['edad_cliente'] > 90, 'edad_cliente'] = np.nan

        # 2. PUNTAJE y PUNTAJE_DATACREDITO: Valores negativos a NaN
        for col in ['puntaje', 'puntaje_datacredito']:
            if col in X_out.columns:
                X_out.loc[X_out[col] < 0, col] = np.nan

        # 3. TENDENCIA_INGRESOS: Limpieza de categorías no válidas
        if 'tendencia_ingresos' in X_out.columns:
            categorias_validas = ['Creciente', 'Decreciente', 'Estable']
            X_out['tendencia_ingresos'] = X_out['tendencia_ingresos'].apply(
                lambda x: x if x in categorias_validas else 'Sin_Informacion'
            )

        # 4. TIPO_LABORAL: Normalización de formato y categorías
        if 'tipo_laboral' in X_out.columns:
            categorias_laborales_validas = ['Empleado', 'Independiente']
            X_out['tipo_laboral'] = X_out['tipo_laboral'].apply(
                lambda x: x if x in categorias_laborales_validas else 'Sin_Informacion'
            )

        # 5. IMPUTACIÓN AGRUPADA POR TIPO_LABORAL (Mediana específica por segmento)
        if 'tipo_laboral' in X_out.columns:
            cols_agrupadas = ['edad_cliente', 'saldo_total', 'saldo_principal', 'promedio_ingresos_datacredito']
            for col in cols_agrupadas:
                if col in X_out.columns:
                    X_out[col] = X_out.groupby('tipo_laboral')[col].transform(
                        lambda x: x.fillna(x.median())
                    )

        return X_out

# 2. Construcción del Pipeline Principal (Transformando dinamicamente las columnas)
class PreprocesadorDinamico(BaseEstimator, TransformerMixin):
    """
    Clase que detecta dinámicamente las columnas numéricas, nominales y ordinales
    presentes en el DataFrame para aplicar la codificación e imputación adecuada.
    """
    def __init__(self, cat_nom_cols=['tipo_laboral', 'tipo_credito'], cat_ord_cols=['tendencia_ingresos'], orden_ordinales=ORDEN_ORDINALES):
        self.cat_nom_cols = cat_nom_cols
        self.cat_ord_cols = cat_ord_cols
        self.orden_ordinales = orden_ordinales
        self.ct = None

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        # 1. Asegurando que las categóricas nominales que ya visualizamos en EDA se traten como texto
        for col in self.cat_nom_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str)

        # 2. Identificando columnas ordinales existentes (tipo_credito)
        ord_existentes = [c for c in self.cat_ord_cols if c in X_copy.columns]

        # 3. Detectando categóricas nominales (incluyendo las forzadas y cualquier object/category/str)
        cat_nom_existentes = list(set([c for c in self.cat_nom_cols if c in X_copy.columns] + 
                                     X_copy.select_dtypes(include=['object', 'category', 'str']).columns.difference(ord_existentes).tolist()))

        # 4. Detectando dinámicamente numéricas reales (excluyendo nominales y ordinales)
        num_cols = X_copy.select_dtypes(include=[np.number]).columns.difference(ord_existentes + cat_nom_existentes).tolist()

        # Pipelines específicos por tipo de dato
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_nom_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Sin_Informacion')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        cat_ord_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Sin_Informacion')),
            ('encoder', OrdinalEncoder(categories=self.orden_ordinales, handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        transformers = []
        if num_cols:
            transformers.append(('num', num_pipeline, num_cols))
        if cat_nom_existentes:
            transformers.append(('cat_nom', cat_nom_pipeline, cat_nom_existentes))
        if ord_existentes:
            transformers.append(('cat_ord', cat_ord_pipeline, ord_existentes))

        self.ct_ = ColumnTransformer(transformers=transformers, remainder='drop')
        self.ct_.fit(X_copy, y)
        return self

    def transform(self, X):
        # Validar que fit() ya fue ejecutado
        check_is_fitted(self, 'ct_')
        
        X_copy = X.copy()
        for col in self.cat_nom_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str)
                
        return self.ct_.transform(X_copy)

# Nuevo transformador para los modelos: Generación de Ratios Financieros para mejorar rendimientos de los modelos
class GeneradorRatiosFinancieros(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Ratio Cuota / Ingreso (Carga financiera mensual)
        if 'cuota_pactada' in X_out.columns and 'salario_cliente' in X_out.columns:
            X_out['ratio_cuota_ingreso'] = X_out['cuota_pactada'] / (X_out['salario_cliente'] + 1)

        # 2. Ratio Saldo Mora / Saldo Total (Severidad de la mora actual)
        if 'saldo_mora' in X_out.columns and 'saldo_total' in X_out.columns:
            X_out['ratio_mora_saldo'] = X_out['saldo_mora'] / (X_out['saldo_total'] + 1)

        # 3. Ratio Capital Prestado / Ingresos (Nivel de endeudamiento del nuevo crédito)
        if 'capital_prestado' in X_out.columns and 'salario_cliente' in X_out.columns:
            X_out['ratio_prestamo_ingreso'] = X_out['capital_prestado'] / (X_out['salario_cliente'] + 1)

        # 4. Total Créditos Vigentes en el Sector (Apalancamiento total)
        cols_sector = ['creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal']
        cols_presentes = [c for c in cols_sector if c in X_out.columns]
        if cols_presentes:
            X_out['total_creditos_sectores'] = X_out[cols_presentes].sum(axis=1)

        return X_out


# 3. Finalización del Pipeline Ensanblado
def construir_pipeline_preprocesamiento(cat_ord_cols=['tendencia_ingresos'], orden_ordinales=ORDEN_ORDINALES):
    """
    Ensamble secuencial:
    1. Limpieza de negocio (reglas EDA)
    2. Creación de Ratios Financieros
    3. Preprocesamiento dinámico (Imputación + Encoders + Scaler)
    """       

    print(" Preprocesamiento realizado con exito ✅") 
    return Pipeline(steps=[
        ('limpieza_inicial', LimpiadorPersonalizado()),
        ('ratios_financieros', GeneradorRatiosFinancieros()),
        ('preprocesamiento_dinamico', PreprocesadorDinamico(cat_ord_cols=cat_ord_cols, orden_ordinales=orden_ordinales))
    ])


# 4. Prueba de Ejecución
if __name__ == "__main__":
    df = cargarDatos()
    
    # Simulación básica: separar la variable objetivo del resto de características
    X = df.drop(columns=['Pago_atiempo'], errors='ignore')
    y = df['Pago_atiempo']

    # Instanciar y probar la transformación
    pipeline = construir_pipeline_preprocesamiento()
    X_procesado = pipeline.fit_transform(X)

    print("¡Módulo ft_engineering validado con éxito!")
    print("Forma de X recibida por el pipeline:", X.shape)
    print("Forma final de la matriz procesada (X):", X_procesado.shape)