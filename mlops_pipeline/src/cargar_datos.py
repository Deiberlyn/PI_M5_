# Importe de Librerias

import os
import pandas as pd

# Se genera la función cargarDatos
def cargarDatos(nombre_archivo="Base_de_datos.xlsx"):
    # 1. Ruta absoluta del directorio src
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # 2. Subir un nivel hacia la carpeta raíz PI_M5_
    ruta_proyecto = os.path.dirname(ruta_actual)

    # 3. Construir la ruta completa al archivo Excel
    ruta_excel = os.path.join(ruta_proyecto, nombre_archivo)

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