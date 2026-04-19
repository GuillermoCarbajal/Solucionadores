import numpy as np 
import pandas as pd

def esta_persona_en_SIV(df_SIV, df_IAE):
    
    df_IAE["SIV_"] = df_IAE["CEDULA"].isin(df_SIV["cedula"]).astype(int)    
    return df_IAE

def get_dosis_vacuna(df_base, df_siv, vacuna, nombre_col):
    df_filtrado = df_siv[df_siv['descripcion'].str.contains(vacuna, na=False)]
    
    df_agg = df_filtrado.groupby('cedula').agg({'dosis': 'last'})
    
    df_base[nombre_col] = df_base['CEDULA'].map(df_agg['dosis'])
    
    return df_base

def agregar_campos_SIV(df_SIV, df_IAE):
    df_IAE = esta_persona_en_SIV(df_SIV, df_IAE)
    df_IAE = get_dosis_vacuna(df_IAE, df_SIV, 'ANTIPOLIOMELÍTICA', 'ANTIPOLIOMELITICA')
    df_IAE = get_dosis_vacuna(df_IAE, df_SIV, 'COVID 19', 'COVID 19')
    return df_IAE