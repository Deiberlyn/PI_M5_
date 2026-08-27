# Importe de Librerias
import os
from pathlib import Path
import pandas as pd

# Se genera la función cargarDatos
def cargarDatos(nombre_archivo="Base_de_datos.xlsx"):
    # 1. Ruta absoluta del directorio src

    # Obtención desde la raíz del proyecto (PI_M5_) subiendo 2 niveles desde src/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    

    # 2. Construyendo la ruta exacta al archivo
    ruta_excel = BASE_DIR / nombre_archivo

    # 3. Leyendo los datos
    df = pd.read_excel(ruta_excel)

    # 4. Leer los datos
    df = pd.read_excel(ruta_excel)
    print(df)
    return df


if __name__ == "__main__":
    datos = cargarDatos()

    print("\n--- PRIMERAS FILAS ---")
    print(datos.head())

    print("\n--- COLUMNAS ENCONTRADAS ---")
    print(datos.columns)