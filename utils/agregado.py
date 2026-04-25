import pandas as pd
import numpy as np

def moda_o_nan(serie):
    moda = serie.mode()
    return moda.iloc[0] if not moda.empty else None

def min_o_nan(serie):
    minimum = serie.min()
    return minimum

def ultimo_no_nulo(serie):    
    return  serie.dropna().iloc[-1] if not serie.dropna().empty else None

def ultimo(serie):    
    return  serie.iloc[-1] if not serie.empty else None

def penultimo_no_nulo(serie):    
    return  serie.dropna().iloc[-2] if len(serie.dropna())>1 else None

def antepenultimo_no_nulo(serie):    
    return  serie.dropna().iloc[-3] if len(serie.dropna())>2 else None

def promedio_entre_intentos(serie):
    s = serie.sort_values()
    # diferencias entre fechas consecutivas
    difs = s.diff().dropna()   # timedeltas
    promedio = difs.mean()     # promedio como timedelta
    promedio = promedio.total_seconds() / 86400 # 86400 segundos en un día

    return promedio

def ultimo_intento(serie):
    s = serie.sort_values()
    ultimo_intento = s.iloc[-1]
    return ultimo_intento

def penultimo_intento(serie):
    s = serie.sort_values()
    penultimo_intento = s.iloc[-2] if len(s)>1 else np.nan
    return penultimo_intento

def antepenultimo_intento(serie):
    s = serie.sort_values()
    antepenultimo_intento = s.iloc[-3] if len(s)>2 else np.nan
    return antepenultimo_intento


def agregar_campo(base, campo, funcion_criterio, nombre_nuevo_atributo):
    if isinstance(funcion_criterio, str):
        funcion_criterio = globals()[funcion_criterio]

    campo_agregado = base.groupby("CEDULA").agg(**{nombre_nuevo_atributo : (campo, funcion_criterio)})
    return campo_agregado


def agregar_base_intentos(df_IAE, dataset):

    if dataset==1:
        campo_nacimiento = 'NACIMIENTO'
        campo_prestador = 'PRESTADOR RECODIFICADO'
        campo_decision = 'DECISIÓN'
        campo_IAE_previo = 'IAE PREVIO'
    else:
        campo_nacimiento = 'FECHA NACIMIENTO'
        campo_prestador = 'PRESTADOR'
        campo_decision = 'DECISION'
        campo_IAE_previo = 'IAE PREVIO'

    agg_dict={}
    agg_dict['Sexo'] = ("PERSONA", moda_o_nan)
    agg_dict['NACIMIENTO'] = (campo_nacimiento, moda_o_nan)
    agg_dict['GRUPO_EDAD_'] = ('GRUPO_EDAD_', ultimo_no_nulo)

    agg_dict['METODO_IAE_FRECUENTE_'] = ("METODO_", moda_o_nan)
    agg_dict['METODO_IAE_PREVIO_'] = ("METODO_", ultimo_no_nulo)
    agg_dict['METODO_IAE_PREVIO_2_'] = ("METODO_", penultimo_no_nulo)
    agg_dict['METODO_IAE_PREVIO_3_'] = ("METODO_", antepenultimo_no_nulo)
    agg_dict['IAE_PREVIO'] = (campo_IAE_previo, ultimo_no_nulo)
    agg_dict['PRESTADOR_RECODIFICADO'] = (campo_prestador, ultimo_no_nulo)
    agg_dict['PRESTADOR_PUBLICO_'] = ("PRESTADOR_PUBLICO_", moda_o_nan)
    agg_dict['PRESTADOR_PRIVADO_'] = ("PRESTADOR_PRIVADO_", moda_o_nan)
    agg_dict['REGISTRO_ultimo_no_nulo'] = ("REGISTRO", ultimo_no_nulo)
    agg_dict['REGISTRO_ultimo'] = ("REGISTRO", ultimo)
    agg_dict['REGISTRO_cantidad'] = ("REGISTRO", 'count')
    agg_dict['FECHA_IAE_ultimo_no_nulo'] = ("FECHA IAE", ultimo_no_nulo)
    agg_dict['FECHA_IAE_ultimo'] = ("FECHA IAE", ultimo)
    agg_dict['FECHA_IAE_cantidad'] = ("FECHA IAE", 'count')

    agg_dict['NUMERO_INTENTOS_'] = ("FECHA IAE", 'count')
    agg_dict['DECISION_'] = (campo_decision, ultimo_no_nulo)
    if dataset==2:
        agg_dict['FECHA SEGUIMIENTO_'] = ('FECHA SEGUIMIENTO_', ultimo)
        agg_dict['FECHA LLAMADA PRESTADOR_'] = ('FECHA LLAMADA PRESTADOR_', ultimo)
        agg_dict['AGENDO CONSULTA ESM 7DIAS SI/NO/INTERNADO_'] = ('AGENDO CONSULTA ESM 7DIAS SI/NO/INTERNADO_', ultimo)
        agg_dict['FECHA DE CONSULTA_'] = ('FECHA DE CONSULTA_', ultimo)
        agg_dict['CONCURRIO_'] = ('CONCURRIO_', ultimo)
        agg_dict['SE LLAMO A USUARIO Y/O REFERENTE_'] = ('SE LLAMO A USUARIO Y/O REFERENTE_', ultimo)
        agg_dict['AGENDO NUEVA CONSULTA_'] = ('AGENDO NUEVA CONSULTA_', ultimo)
        agg_dict['FECHA DE NUEVA CONSULTA_'] = ('FECHA DE NUEVA CONSULTA_', ultimo)
        agg_dict['INTERNACION_'] = ('INTERNACION_', ultimo)
        agg_dict['FECHA ALTA_'] = ('FECHA ALTA_', ultimo)
        agg_dict['FECHA LLAMADA PRESTADOR.1_'] = ('FECHA LLAMADA PRESTADOR.1_', ultimo)
        agg_dict['MSP_'] = ('MSP_', ultimo)
        agg_dict['FECHA DE LLAMADA AL PRESTADOR_'] = ('FECHA DE LLAMADA AL PRESTADOR_', ultimo)
        agg_dict['CONCURRIO_binaria'] = ('CONCURRIO_binaria', ultimo)
        agg_dict['NO_CONCURRIO_CONSULTA_'] = ('NO_CONCURRIO_CONSULTA_', ultimo)



    agg_dict['ULTIMO_INTENTO_'] = ("FECHA IAE", ultimo_intento)
    agg_dict['DIAS_PROMEDIO_INTENTOS_'] = ("FECHA IAE", promedio_entre_intentos)
    agg_dict['MIN_DIAS_IAE_MUERTE_'] = ("DIAS_IAE_MUERTE_", min_o_nan)

    agg_dict['IAE_en_CDE_'] = ('IAE_en_CDE', ultimo_no_nulo)
    agg_dict['FECHA_DEFUNCION'] = ("FECHA_DEFUNCION", ultimo_no_nulo)
    agg_dict['CAT_SUI_'] = ('CAT_SUI_', ultimo_no_nulo)
    agg_dict['CAT_MCEXSUI_'] = ('CAT_MCEXSUI_', ultimo_no_nulo)


    if dataset==2:
        agg_dict['MOTIVO_EXT_SUI_'] = ('MOTIVO_EXT_SUI_', moda_o_nan)
        agg_dict['MOTIVO_EXTERNO_'] = ('MOTIVO_EXTERNO_', moda_o_nan)
        agg_dict['ES_MOTIVO_EXTERNO_'] = ('ES_MOTIVO_EXTERNO_', moda_o_nan)
    
    agg_dict['CNV_'] = ("CNV_", moda_o_nan)
    
    agg_dict['CNV_ultimo_cant_hijos_'] = ("CNV_cant_hijos_cuando_IAE", ultimo_no_nulo)
    agg_dict['CNV_ultimo_edad_hijo_mayor_'] = ("CNV_edad_hijo_mayor_cuando_IAE", ultimo_no_nulo)
    agg_dict['CNV_ultimo_edad_hijo_menor_'] = ("CNV_edad_hijo_menor_cuando_IAE", ultimo_no_nulo)

    if dataset==2:
        agg_dict['CNV_nro_rese_in_IAE_'] = ("CNV_nro_rese_in_IAE_", moda_o_nan)
        agg_dict['CNV_otro_progenitor_'] = ("CNV_otro_progenitor_", moda_o_nan)
        agg_dict['CNV_ultimo_estado_civil_'] = ("CNV_ultimo_cuando_IAE_estado_civil", ultimo_no_nulo)
        agg_dict['CNV_ultimo_pais_nac_'] = ("CNV_ultimo_cuando_IAE_pais_nac", ultimo_no_nulo)
        agg_dict['CNV_ultimo_pais_residencia_'] = ("CNV_ultimo_cuando_IAE_pais_residencia", ultimo_no_nulo)
        agg_dict['CNV_ultimo_mayor_nivel_estudio_'] = ("CNV_ultimo_cuando_IAE_mayor_nivel_estudio", ultimo_no_nulo)
        agg_dict['CNV_ultimo_sexo_'] = ("CNV_ultimo_cuando_IAE_sexo", ultimo_no_nulo)
        agg_dict['CNV_ultimo_embarazo_anteriores_'] = ("CNV_ultimo_cuando_IAE_numero_embarazo_anteriores", ultimo_no_nulo)
        agg_dict['CNV_ultimo_semana_embarazo_primer_consulta_'] = ("CNV_ultimo_cuando_IAE_semana_embarazo_primer_consulta", ultimo_no_nulo)
        agg_dict['CNV_ultimo_total_consultas_'] = ("CNV_ultimo_cuando_IAE_total_consultas", ultimo_no_nulo)
    
    agg_dict['RUCAF_prestador'] = ("RUCAF_prestador", ultimo_no_nulo)
    agg_dict['RUCAF_PRESTADOR_PUBLICO_'] = ("RUCAF_PRESTADOR_PUBLICO_", ultimo_no_nulo)
    agg_dict['RUCAF_PRESTADOR_PRIVADO_'] = ("RUCAF_PRESTADOR_PRIVADO_", ultimo_no_nulo)
    agg_dict['RUCAF_pais'] = ("RUCAF_pais", ultimo_no_nulo)
    agg_dict['RUCAF_departamento'] = ("RUCAF_departamento", ultimo_no_nulo)
    agg_dict['RUCAF_localidad'] = ("RUCAF_localidad", ultimo_no_nulo)
    agg_dict['RUCAF_cobertura'] = ("RUCAF_cobertura", ultimo_no_nulo)

    agg_dict['SHARPS_'] = ("SHARPS_", ultimo_no_nulo)

    agg_dict['SIV_'] = ("SIV_", ultimo_no_nulo)
    agg_dict['ANTIPOLIOMELITICA'] = ("ANTIPOLIOMELITICA", ultimo_no_nulo)
    agg_dict['COVID 19'] = ("COVID 19", ultimo_no_nulo)



        

    
    df_IAE_agregada = df_IAE.groupby("CEDULA").agg(**agg_dict).reset_index()

    #df_IAE_procesada = (
    #    df_IAE.groupby("CEDULA")
    #    .agg(
    #        Sexo=,
    #        NACIMIENTO=(campo_nacimiento, moda_o_nan),
    #        METODO_IAE_FRECUENTE_=("METODO_", moda_o_nan),
    #        METODO_IAE_PREVIO_=("METODO_", ultimo_no_nulo),
    #        METODO_IAE_PREVIO_2=("METODO_", penultimo_no_nulo),
    #        METODO_IAE_PREVIO_3=("METODO_", antepenultimo_no_nulo),
    #        IAE_PREVIO=("IAE PREVIO", ultimo_no_nulo),
    #        PRESTADOR_RECODIFICADO=(campo_prestador, ultimo_no_nulo),
    #        PRESTADOR_PUBLICO_=("PRESTADOR_PUBLICO_", moda_o_nan),
    #        PRESTADOR_PRIVADO_=("PRESTADOR_PRIVADO_", moda_o_nan),
    #        REGISTRO=("REGISTRO", ultimo_no_nulo),
    #        FECHA_IAE=("FECHA IAE", ultimo_no_nulo),
    #        FECHA_DEFUNCION=("FECHA_DEFUNCION", ultimo_no_nulo),
    #        NUMERO_INTENTOS_=("FECHA IAE", "count"),
    #        DECISION_=(campo_decision, ultimo_no_nulo),
    #        DEFUNCION_=("DEFUNCION_", ultimo_no_nulo),
    #        CAT_SUI_=("CAT_SUI_", ultimo_no_nulo),
    #        CAT_MCEXSUI_=("CAT_MCEXSUI_", ultimo_no_nulo),
    #        MOTIVO_EXT_SUI_=("MOTIVO_EXT_SUI_",ultimo_no_nulo),
    #        GRUPO_EDAD_=("GRUPO_EDAD_", ultimo_no_nulo),
    #        ULTIMO_INTENTO_=("FECHA IAE", ultimo_intento),
    #        DIAS_PROMEDIO_INTENTOS_=("FECHA IAE", promedio_entre_intentos),
    #        MIN_DIAS_IAE_MUERTE_=("DIAS_IAE_MUERTE_", min_o_nan),
    #    )
    #    .reset_index()

    return df_IAE_agregada

def personas_con_IAE_no_presentes_en_CDE(df_IAE_agregada, df_IAE_CDE):
    
    indices = df_IAE_agregada['CEDULA'].isin(df_IAE_CDE['cedula'])
    personas_de_IAE_no_presentes_en_CDE = df_IAE_agregada[~indices]
    print(f'Hay {personas_de_IAE_no_presentes_en_CDE.shape[0]} personas con IAE no presentes en CDE.')
    
    return personas_de_IAE_no_presentes_en_CDE


def personas_en_CDE_sin_IAE(df_IAE_agregada, df_IAE_CDE):
    
    indices = df_IAE_CDE['cedula'].isin(df_IAE_agregada['CEDULA'])
    personas = df_IAE_CDE[~indices]
    print(f'Hay {personas.shape[0]} personas en CDE que no están en la base de IAE.')
    
    return personas

def agregar_IAE_PREVIO_corregido(df_IAE_agregada):

    df_IAE_agregada['IAE_PREVIO_CORREGIDO' ] = df_IAE_agregada['IAE_PREVIO'].copy()
    df_IAE_agregada.loc[(df_IAE_agregada["IAE_PREVIO_CORREGIDO"] == "NO SE INDICA") & (df_IAE_agregada["NUMERO_INTENTOS_"] > 1), "IAE_PREVIO_CORREGIDO"] = "SI"
    
    return df_IAE_agregada

def personas_con_IAE_no_presentes_en_CNV(df_IAE_agregada, df_IAE_CNV):
    
    indices = df_IAE_agregada['CEDULA'].isin(df_IAE_CNV['cedula'])
    personas_de_IAE_no_presentes_en_CNV = df_IAE_agregada[~indices]
    print(f'Hay {personas_de_IAE_no_presentes_en_CNV.shape[0]} personas con IAE no presentes en CNV.')
    
    return personas_de_IAE_no_presentes_en_CNV

def personas_en_CNV_sin_IAE(df_IAE_agregada, df_IAE_CNV):
    
    indices = df_IAE_CNV['cedula'].isin(df_IAE_agregada['CEDULA'])
    personas = df_IAE_CNV[~indices]
    print(f'Hay {personas.shape[0]} personas en CNV que no están en la base de IAE.')
    
    return personas

def personas_con_IAE_no_presentes_en_RUCAF(df_IAE_agregada, df_IAE_RUCAF):
    
    indices = df_IAE_agregada['CEDULA'].isin(df_IAE_RUCAF['cedula'])
    personas_de_IAE_no_presentes_en_RUCAF = df_IAE_agregada[~indices]
    print(f'Hay {personas_de_IAE_no_presentes_en_RUCAF.shape[0]} personas con IAE no presentes en RUCAF.')
    
    return personas_de_IAE_no_presentes_en_RUCAF

def personas_en_RUCAF_sin_IAE(df_IAE_agregada, df_IAE_RUCAF):
    
    indices = df_IAE_RUCAF['cedula'].isin(df_IAE_agregada['CEDULA'])
    personas = df_IAE_RUCAF[~indices]
    print(f'Hay {personas.shape[0]} personas en RUCAF que no están en la base de IAE.')
    
    return personas

def personas_con_IAE_no_presentes_en_SHARPS(df_IAE_agregada, df_IAE_SHARPS):
    
    indices = df_IAE_agregada['CEDULA'].isin(df_IAE_SHARPS['cedula'])
    personas_de_IAE_no_presentes_en_SHARPS = df_IAE_agregada[~indices]
    print(f'Hay {personas_de_IAE_no_presentes_en_SHARPS.shape[0]} personas con IAE no presentes en SHARPS.')
    
    return personas_de_IAE_no_presentes_en_SHARPS

def personas_en_SHARPS_sin_IAE(df_IAE_agregada, df_IAE_SHARPS):
    
    indices = df_IAE_SHARPS['cedula'].isin(df_IAE_agregada['CEDULA'])
    personas = df_IAE_SHARPS[~indices]
    print(f'Hay {personas.shape[0]} personas en SHARPS que no están en la base de IAE.')
    
    return personas

def personas_con_IAE_no_presentes_en_SIV(df_IAE_agregada, df_IAE_SIV):
    
    indices = df_IAE_agregada['CEDULA'].isin(df_IAE_SIV['cedula'])
    personas_de_IAE_no_presentes_en_SIV = df_IAE_agregada[~indices]
    print(f'Hay {personas_de_IAE_no_presentes_en_SIV.shape[0]} personas con IAE no presentes en SIV.')
    
    return personas_de_IAE_no_presentes_en_SIV

def personas_en_SIV_sin_IAE(df_IAE_agregada, df_IAE_SIV):
    
    indices = df_IAE_SIV['cedula'].isin(df_IAE_agregada['CEDULA'])
    personas = df_IAE_SIV[~indices]
    print(f'Hay {personas.shape[0]} personas en SIV que no están en la base de IAE.')
    
    return personas