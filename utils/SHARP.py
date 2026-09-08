import numpy as np 
import pandas as pd

def esta_persona_en_SHARPS(df_SHARPS, df_IAE):
    
    if df_SHARPS.empty:
        df_IAE["SHARPS_"] = False
    else:
        df_IAE["SHARPS_"] = df_IAE["CEDULA"].isin(
            df_SHARPS["cedula"]
        )
    
    return df_IAE