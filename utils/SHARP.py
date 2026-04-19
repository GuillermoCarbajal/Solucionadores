import numpy as np 
import pandas as pd

def esta_persona_en_SHARPS(df_SHARPS, df_IAE):
    
    df_IAE["SHARPS_"] = df_IAE["CEDULA"].isin(df_SHARPS["cedula"]).astype(int)    
    return df_IAE