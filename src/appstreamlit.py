# Librerias base manejo de datos
import pandas as pd

# Librerias para graficos
import plotly.express as px
import plotly.graph_objects as go

# Librerias para gestion de datos y modelos
from sklearn.model_selection import train_test_split

# Libreria manejo app
import streamlit as st

# Funciones generadas en pipelines anteriores
from cargar_datos import cargarDatos
from model_monitoring import evaluar_drift_dataset

#####################################################################
###### PRUEBA PIPELINE MODEL_MONITORIND ANTES DE CORRER LA APP ######
#####################################################################
if __name__ == "__main__":
    df = cargarDatos()
    df_ref, df_new = train_test_split(df, test_size=0.2, random_state=42)
    reporte = evaluar_drift_dataset(df_ref, df_new)
    print(reporte)

#####################################################################
################ Producción y gestión de Streamlit ##################
#####################################################################


# 1. Configuración de la interfaz de la pág, streamlit
st.set_page_config(
    page_title="Dashboard de Monitoreo de Data Drift",
    page_icon="🛃",
    layout="wide"
)

st.title("📊 Dashboard de Monitoreo y Data Drift")
st.markdown("Evaluación continua de la distribución de datos crudos (Histórico vs. Actual)")

# 2. ------------ Carga de Datos
@st.cache_data
def obtener_datos():
    df = cargarDatos()
    df_ref, df_new = train_test_split(df, test_size=0.2, random_state=42)
    return df_ref, df_new

df_ref, df_new = obtener_datos()
reporte_drift = evaluar_drift_dataset(df_ref, df_new)

# 3.------------- Visualización de Métricas & Semáforos
st.subheader("🚦 Visualización de Métricas y Estado General")

col1, col2, col3 = st.columns(3)
total_vars = len(reporte_drift)
estables = len(reporte_drift[reporte_drift['Estado'] == 'ESTABLE'])
criticas = len(reporte_drift[reporte_drift['Estado'] == 'CRÍTICO'])

col1.metric("Total Variables Monitoreadas", total_vars)
col2.metric("Variables Estables", estables, delta="Sin Drift")
col3.metric("Variables Críticas", criticas, delta="- Alerta", delta_color="inverse")

# 3.1 Puliendo tablas métricas
def resaltar_estado(val):
    if val == 'CRÍTICO':
        return 'background-color: #ff4b4b; color: white; font-weight: bold;'
    elif val == 'ADVERTENCIA':
        return 'background-color: #ffa726; color: black; font-weight: bold;'
    return 'background-color: #66bb6a; color: white; font-weight: bold;'

st.dataframe(
    reporte_drift.style.map(resaltar_estado, subset=['Estado']),
    use_container_width=True
)

# 3.2 Comparación gráfica por variable
st.markdown("#### Comparación (simulación) de Distribución: Histórico vs. Actual")
var_seleccionada = st.selectbox("Selecciona una variable para inspeccionar:", df_ref.columns)
fig = go.Figure()
if pd.api.types.is_numeric_dtype(df_ref[var_seleccionada]):
    fig.add_trace(go.Histogram(x=df_ref[var_seleccionada], name="Histórico (Ref)", opacity=0.6, marker_color="blue"))
    fig.add_trace(go.Histogram(x=df_new[var_seleccionada], name="Actual (New)", opacity=0.6, marker_color="orange"))
    fig.update_layout(barmode='overlay', title=f"Distribución de {var_seleccionada}")
else:
    df_ref_counts = df_ref[var_seleccionada].astype(str).value_counts(normalize=True).reset_index()
    df_new_counts = df_new[var_seleccionada].astype(str).value_counts(normalize=True).reset_index()
    df_ref_counts.columns = ['Categoría', 'Proporción']
    df_new_counts.columns = ['Categoría', 'Proporción']
    df_ref_counts['Origen'] = 'Histórico (Ref)'
    df_new_counts['Origen'] = 'Actual (New)'
    comp_df = pd.concat([df_ref_counts, df_new_counts])
    fig = px.bar(comp_df, x='Categoría', y='Proporción', color='Origen', barmode='group', title=f"Distribución Categorías: {var_seleccionada}")

st.plotly_chart(fig, use_container_width=True)

st.divider()

# 4. ---------- Análisis Temporal
st.subheader("⏳ Análisis Temporal")

if 'fecha_prestamo' in df_ref.columns:
    df_temp = df_new.copy()
    df_temp['fecha_prestamo'] = pd.to_datetime(df_temp['fecha_prestamo'], errors='coerce')
    df_temp = df_temp.dropna(subset=['fecha_prestamo'])
    
    # Agrupamiento por mes o fecha para ver tendencias
    conteo_temporal = df_temp.set_index('fecha_prestamo').resample('ME').size().reset_index(name='Volumen_Registros')
    
    fig_temp = px.line(
        conteo_temporal, 
        x='fecha_prestamo', 
        y='Volumen_Registros', 
        markers=True,
        title="Evolución del Volumen de Datos en el Tiempo (Detección de Cambios Abruptos)"
    )
    st.plotly_chart(fig_temp, use_container_width=True)
else:
    st.info("No se encontró la columna de tiempo 'fecha_prestamo' para generar la tendencia temporal.")

st.divider()

# 5. ---------- RECOMENDACIONES
st.subheader("🟡 Recomendaciones del Sistema")
if criticas > 0:
    st.error(f"⚠️ **¡Alerta Crítica!** Se ha detectado Data Drift en {criticas} variable(s).")
    st.markdown("""
    * **Acción sugerida:** Ejecutar pipeline de reentrenamiento (*retraining*) del modelo.
    * **Revisión de Variables:** Inspeccionar las fuentes de ingesta para validar posibles cambios de formato o sesgos en la captura de datos.
    """)
else:
    st.success("✅ **Sistema Estable:** No se detectan desviaciones significativas entre las distribuciones.")
    st.markdown("""
    * **Acción sugerida:** Mantener el modelo actual en producción.
    * **Próximo ciclo:** Continuar el monitoreo en el siguiente lote de inferencia.
    """)