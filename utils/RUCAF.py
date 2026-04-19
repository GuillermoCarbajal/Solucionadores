import numpy as np 
import pandas as pd

def agregar_datos_RUCAF_en_IAE(df_RUCAF, df_IAE):
    
    df_IAE_RUCAF_grouped = df_RUCAF.groupby('cedula').last()
    df_IAE_RUCAF_grouped = df_IAE_RUCAF_grouped.reset_index()

    df_IAE["RUCAF_prestador"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["prestador"])
    df_IAE["RUCAF_pais"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["pais"])
    df_IAE["RUCAF_departamento"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["departamento"])
    df_IAE["RUCAF_localidad"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["localidad"])
    df_IAE["RUCAF_cobertura"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["cobertura"])

    return df_IAE