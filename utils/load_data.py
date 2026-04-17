import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt

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