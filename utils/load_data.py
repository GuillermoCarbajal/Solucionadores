import pandas as pd 
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import pandas as pd
import os

def load_excelfile(filename):
    '''
    Entrada:
        filename: name of the excel file
    Salida:
        dfs: diccionario que contiene un data frame por cada hoja en la planilla
    '''
    # Crear un objeto ExcelFile
    xls = pd.ExcelFile(filename, engine="openpyxl")

    # Ver los nombres de las hojas
    print(xls.sheet_names)

    # Cargar cada hoja como un DataFrame
    dfs = {nombre: xls.parse(nombre) for nombre in xls.sheet_names}
    return dfs

def load_databases(filename, entrega=1):
    dfs = load_excelfile(filename)
    #segunda_entrega = 'EH' in dfs.keys()

    ## Ejemplo: acceder a una hoja
    df_IAE = dfs["IAE"]
    df_IAE_CDE = dfs["IAE_CDE"] if 'IAE_CDE' in dfs.keys() else dfs["CDE"]
    df_IAE_CNV = dfs["IAE_CNV"] if 'IAE_CNV' in dfs.keys() else dfs["CNV"]
    df_IAE_RUCAF = dfs["IAE_RUCAF"] 
    df_IAE_SHARPS = dfs["IAE_SHARPS"] if 'IAE_SHARPS' in dfs.keys() else dfs["SHARPS"]
    df_IAE_SIV = dfs["IAE_SIV"] if 'IAE_SIV' in dfs.keys() else dfs["SIV"]
    df_IAE_EH = dfs["EH"] if 'EH' in dfs.keys() else None
    
    if entrega==2:
        output = df_IAE, df_IAE_CDE, df_IAE_CNV, df_IAE_RUCAF, df_IAE_SHARPS, df_IAE_SIV, df_IAE_EH
    elif entrega==1:
        output = df_IAE, df_IAE_CDE, df_IAE_CNV, df_IAE_RUCAF, df_IAE_SHARPS, df_IAE_SIV
    else:
        print('Indique que entrega quiere levantar')
    
    return output



def cargar_datos(directorio):
    bases = {}

    for archivo in sorted(os.listdir(directorio)):
        if not archivo.lower().endswith(".csv"):
            continue

        ruta = os.path.join(directorio, archivo)
        print(f"Cargando {archivo}...")

        filas_validas = []
        filas_invalidas = []

        with open(ruta, "r", encoding="utf-8-sig", errors="replace") as f:
            encabezado = f.readline().rstrip("\n\r")
            columnas = encabezado.split(";")
            n_columnas = len(columnas)

            for num_linea, linea in enumerate(f, start=2):
                campos = linea.rstrip("\n\r").split(";")

                if len(campos) == n_columnas:
                    filas_validas.append(campos)
                else:
                    filas_invalidas.append((num_linea, len(campos)))

        if filas_invalidas:
            print(
                f"  {len(filas_invalidas)} filas no leídas "
                f"(se esperaban {n_columnas} columnas)."
            )

            # Mostrar las primeras 10 para poder inspeccionarlas
            print("  Primeras filas problemáticas:")
            for num_linea, n_campos in filas_invalidas[:10]:
                print(f"    Línea {num_linea}: {n_campos} campos")

        df = pd.DataFrame(filas_validas, columns=columnas)

        nombre = os.path.splitext(archivo)[0]
        bases[nombre] = df

        print(f"  {len(filas_validas)} filas cargadas.")

    return bases


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.dump(obj, f)