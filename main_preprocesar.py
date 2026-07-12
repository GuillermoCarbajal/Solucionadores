import numpy as np
import pandas as pd
from utils.utils import discretizar, conservar_filas_con_n_no_nulos, get_filas_con_n_nulos, eliminar_duplicados, convertir_enteros_a_fecha, eliminar_sin_cedula
from matplotlib import pyplot as plt

from utils.load_data import load_excelfile, load_databases
from utils.estadisticas import mostrar_frecuencias, mostrar_unicos, mostrar_rango, mostrar_valores_nulos, \
                               mostrar_duplicados, graficar_segun_fecha

from utils.IAE import agregar_tipo_prestador_IAE, guardar_prestadores, acondicionar_atributo_metodo, \
                      agregar_categoria_metodo, acondicionar_IAE_PREVIO, calcular_y_agregar_campo_edad, acondicionar_campo_DECISION, \
                      agregar_campo_DECISION, agregar_si_intentos_en_CDE, corregir_fechas_enteras_prestadores, acondicionar_campos_prestadores, \
                      acondicionar_CONCURRIO, acondicionar_campo_agendo_consulta_en_7dias, agregar_si_es_IAE, agregar_si_tiene_fecha_registro, \
                      agregar_fecha_IAE_siguiente, agregar_tiempo_reincidencia, agregar_info_intentos_previos
 
from utils.CDE import conciden_fechas_nacimiento_digitadas_y_calculadas, agregar_datos_CDE_en_IAE, \
                      agregar_atributo_CAT_SUI, argegar_atributo_CAT_MCEXSUI
from utils.agregado import agregar_base_intentos, agregar_campo, personas_con_IAE_no_presentes_en_CDE, personas_en_CDE_sin_IAE, \
                           personas_con_IAE_no_presentes_en_CNV, personas_en_CNV_sin_IAE, \
                        personas_con_IAE_no_presentes_en_RUCAF, personas_en_RUCAF_sin_IAE, personas_con_IAE_no_presentes_en_SHARPS, \
                        personas_en_SHARPS_sin_IAE, personas_con_IAE_no_presentes_en_SIV, personas_en_SIV_sin_IAE

from utils.CNV import esta_persona_en_CNV, agregar_datos_hijos_cuando_intento, agregar_datos_CNV_cuando_intento, \
                      incluir_otro_progrenitor, esta_persona_en_CNV_nro_rese
from utils.RUCAF import agregar_datos_RUCAF_en_IAE, agregar_RUCAF_region
from utils.SHARP import esta_persona_en_SHARPS
from utils.SIV import agregar_campos_SIV    
from utils.EH import agregar_datos_EH_cuando_intento

from datetime import datetime
import argparse

def preprocesar(args):

    # configuración
    dataset = 2 # 1 primera entrega, 2 segunda entrega

    campo_decision = 'DECISION' if dataset==2 else 'DECISIÓN'
    campo_nacimiento = 'FECHA NACIMIENTO' if dataset==2 else 'NACIMIENTO'
    campo_prestador = 'PRESTADOR' if dataset==2 else 'PRESTADOR RECODIFICADO'
    campo_edad = 'EDAD_' if dataset==2 else 'EDAD' 

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    archivo = args.path

    print('Levantando las bases ...')
    # Levantar las bases. La segunda entrega tiene la base de egresos hospitalarios que no tiene la primera
    if dataset ==1:
        df_IAE, df_IAE_CDE, df_IAE_CNV, df_IAE_RUCAF, df_IAE_SHARPS, df_IAE_SIV = load_databases(archivo, dataset)
    elif dataset==2:
        df_IAE, df_IAE_CDE, df_IAE_CNV, df_IAE_RUCAF, df_IAE_SHARPS, df_IAE_SIV, df_IAE_EH = load_databases(archivo, dataset)

    
    print('Eliminando filas de base CDE con menos de 4 columnas no nulas')
    df_IAE_CDE = conservar_filas_con_n_no_nulos(df_IAE_CDE, n=4)        

    df_IAE_CDE = agregar_atributo_CAT_SUI(df_IAE_CDE)
    df_IAE_CDE = argegar_atributo_CAT_MCEXSUI(df_IAE_CDE)

    edades_digitadas = df_IAE_CDE["edad_fallecimiento_digitada"]
    mostrar_rango(df_IAE_CDE,"edad_fallecimiento_digitada")
    mostrar_valores_nulos(df_IAE_CDE,"edad_fallecimiento_digitada")

    edades_calculadas = df_IAE_CDE["edad_fallecimiento_calculada"]
    mostrar_rango(df_IAE_CDE,"edad_fallecimiento_calculada")
    mostrar_valores_nulos(df_IAE_CDE,"edad_fallecimiento_calculada")
    conciden_fechas_nacimiento_digitadas_y_calculadas(df_IAE_CDE)


    # Agrupar edades de a 5 
    df_IAE_CDE = discretizar(df_IAE_CDE, "edad_fallecimiento_calculada", "grupo edades_", 5)

    df_IAE = agregar_si_tiene_fecha_registro(df_IAE)
    df_IAE = agregar_si_es_IAE(df_IAE)


    df_IAE = eliminar_sin_cedula(df_IAE)   
    df_IAE = eliminar_duplicados(df_IAE)    
    df_IAE = agregar_tipo_prestador_IAE(df_IAE, campo_prestador)

    df_IAE = acondicionar_atributo_metodo(df_IAE)

    df_IAE = agregar_categoria_metodo(df_IAE)
    mostrar_frecuencias(df_IAE, 'METODO_')

    df_IAE = acondicionar_IAE_PREVIO(df_IAE)
    mostrar_frecuencias(df_IAE,'IAE PREVIO')
    mostrar_unicos(df_IAE,'IAE PREVIO')

    df_IAE = calcular_y_agregar_campo_edad(df_IAE, 'FECHA IAE', campo_nacimiento)
    mostrar_rango(df_IAE, 'EDAD_')

    df_IAE = discretizar(df_IAE, campo_edad, "GRUPO_EDAD_", 5)

    df_IAE = acondicionar_campo_DECISION(df_IAE, campo_decision)
    mostrar_unicos(df_IAE, campo_decision)
    mostrar_frecuencias(df_IAE, campo_decision)

    agregar_campo_DECISION(df_IAE, campo_decision, 'DECISION_')
    mostrar_unicos(df_IAE,campo_decision)
    mostrar_frecuencias(df_IAE,campo_decision)
    mostrar_unicos(df_IAE, 'DECISION_')
    mostrar_frecuencias(df_IAE, 'DECISION_')

    df_IAE = agregar_si_intentos_en_CDE(df_IAE, df_IAE_CDE, 'IAE_en_CDE') 

    df_IAE = agregar_datos_CDE_en_IAE(df_IAE, df_IAE_CDE, dataset)

    if dataset==2:
        df_IAE = corregir_fechas_enteras_prestadores(df_IAE)
        df_IAE = acondicionar_campos_prestadores(df_IAE)
        df_IAE = acondicionar_CONCURRIO(df_IAE, 'CONCURRIO_', 'CONCURRIO_binaria')
        df_IAE = acondicionar_campo_agendo_consulta_en_7dias(df_IAE,'AGENDO CONSULTA ESM 7DIAS SI/NO/INTERNADO_')
        mostrar_frecuencias(df_IAE,'AGENDO CONSULTA ESM 7DIAS SI/NO/INTERNADO_')

        print('Campo CONCURRIO crudo:', df_IAE['CONCURRIO'].unique().tolist())
        df_IAE['CONCURRIO_'] = convertir_enteros_a_fecha(df_IAE['CONCURRIO'])
        print('Campo CONCURRIO_ luego de convertir enteros a fechas:',df_IAE['CONCURRIO_'].unique().tolist())        
    
        print(df_IAE['AGENDO NUEVA CONSULTA'].unique().tolist())
        df_IAE['AGENDO NUEVA CONSULTA_'] = convertir_enteros_a_fecha(df_IAE['AGENDO NUEVA CONSULTA'])
        print(df_IAE['AGENDO NUEVA CONSULTA_'].unique().tolist())

    eliminar_descartadas_por_rastreador = True
    if eliminar_descartadas_por_rastreador:
        n_antes = df_IAE.shape[0]
        print('Antes de eliminar los intentos descartados por rastreador habían: ', n_antes)
        df_IAE = df_IAE[df_IAE['DESCARTADA_POR_RASTREADOR']==False]
        n_despues = df_IAE.shape[0]
        print('Luego de eliminar los intentos descartados por rastreador hay: ', n_despues)
        print('Se eliminaron ', n_antes-n_despues, 'intentos descartados por los rastreadores')   


    print('Agregando info de intentos anteriores')
    df_IAE = agregar_info_intentos_previos(df_IAE)

    print('Agregando info de intentos posteriores')
    df_IAE = agregar_fecha_IAE_siguiente(df_IAE)
    df_IAE = agregar_tiempo_reincidencia(df_IAE)


    ## CNV ####
    df_IAE_CNV = eliminar_duplicados(df_IAE_CNV)  
    df_IAE_CNV_agrupada = df_IAE_CNV.groupby('cedula')     

    df_IAE = esta_persona_en_CNV(df_IAE_CNV, df_IAE)

    if dataset==2:
        df_IAE = esta_persona_en_CNV_nro_rese(df_IAE_CNV, df_IAE) 

    df_IAE = df_IAE.apply( lambda row: agregar_datos_hijos_cuando_intento(row, df_IAE_CNV_agrupada), axis=1)    

    if dataset==2:
        df_IAE = df_IAE.apply( lambda row: agregar_datos_CNV_cuando_intento(row, df_IAE_CNV_agrupada), axis=1)
        df_IAE=incluir_otro_progrenitor(df_IAE,df_IAE_CNV)

    if dataset==2:
        mostrar_valores_nulos(df_IAE,'persona_en_CNV_nro_rese_')
        mostrar_unicos(df_IAE,'persona_en_CNV_nro_rese_')
        mostrar_frecuencias(df_IAE,'persona_en_CNV_nro_rese_')    

        mostrar_valores_nulos(df_IAE,'CNV_otro_progenitor_')
        mostrar_unicos(df_IAE,'CNV_otro_progenitor_')
        mostrar_frecuencias(df_IAE,'CNV_otro_progenitor_')

    ################ RUCAF #####################
    print('Procesando RUCAF...')
    df_IAE_RUCAF = conservar_filas_con_n_no_nulos(df_IAE_RUCAF, 2)
    df_IAE_RUCAF = eliminar_duplicados(df_IAE_RUCAF)
    df_IAE_RUCAF = agregar_RUCAF_region(df_IAE_RUCAF)
    df_IAE_RUCAF = agregar_tipo_prestador_IAE(df_IAE_RUCAF,'prestador','tipo_prestador_RUCAF')
    personas_en_RUCAF_sin_IAE(df_IAE, df_IAE_RUCAF)
    personas_con_IAE_no_presentes_en_RUCAF(df_IAE, df_IAE_RUCAF)

    df_IAE_RUCAF_grouped = df_IAE_RUCAF.groupby('cedula')
    df_IAE = df_IAE.apply( lambda row: agregar_datos_RUCAF_en_IAE(row, df_IAE_RUCAF_grouped), axis=1)


    ################# SHARPS  ######################
    print('Procesando SHARPS...')
    df_IAE_SHARPS = conservar_filas_con_n_no_nulos(df_IAE_SHARPS, 2)
    personas_con_IAE_no_presentes_en_SHARPS(df_IAE, df_IAE_SHARPS)
    df_IAE = esta_persona_en_SHARPS(df_IAE_SHARPS, df_IAE)

    ################  SIV   #######################
    print('Procesando SIV...')
    df_IAE_SIV = conservar_filas_con_n_no_nulos(df_IAE_SIV, 2)
    df_IAE_SIV = eliminar_duplicados(df_IAE_SIV)
    personas_en_SIV_sin_IAE(df_IAE, df_IAE_SIV)
    df_IAE = agregar_campos_SIV(df_IAE_SIV, df_IAE)

    ################  EH  #########################
    print('Procesando EH...')
    if dataset==2:
        df_IAE_EH_agrupada = df_IAE_EH.groupby('cedula')
        df_IAE = df_IAE.apply( lambda row: agregar_datos_EH_cuando_intento(row, df_IAE, df_IAE_EH_agrupada), axis=1)

    ## Guardar la base de datos procesada pero sin agregar por persona
    nombre_procesada = f'IAE_sin_agregar_entrega{dataset}_{timestamp}.csv' 
    df_IAE.to_csv(nombre_procesada)        

    ############## Agregar por persona    #############################
    print('Agregando los datos por persona...')

    df_IAE_agregada = agregar_base_intentos(df_IAE, dataset)
    agregar_campo(df_IAE,'METODO','ultimo_intento','ULTIMO_INTENTO_')
    personas = personas_con_IAE_no_presentes_en_CNV(df_IAE_agregada, df_IAE_CNV)    


    personas_con_IAE_no_presentes_en_SIV(df_IAE_agregada, df_IAE_SIV)
    personas_con_IAE_no_presentes_en_CDE(df_IAE_agregada, df_IAE_CDE)
    personas_en_CDE_sin_IAE(df_IAE_agregada, df_IAE_CDE)

    #df_IAE_agregada = agregar_IAE_PREVIO_corregido(df_IAE_agregada)

    for atributo in df_IAE_agregada.keys():
        mostrar_valores_nulos(df_IAE_agregada,atributo)
        mostrar_unicos(df_IAE_agregada,atributo)
        mostrar_frecuencias(df_IAE_agregada,atributo) 
    
       
    #########  Guardar la base de datos agregada por persona
    nombre_procesada = f'IAE_agregada_entrega{dataset}_{timestamp}.csv' 
    df_IAE_agregada.to_csv(nombre_procesada)     
  
    return      

def parseCommandLineArguments():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--path', type=str, default='/home/carbajal/Documents/SaludMental/2da entrega 20260210/Planilla completa.xlsx')
    
          
    args = parser.parse_args() 
    return args


if __name__ == "__main__":


    args = parseCommandLineArguments()

    preprocesar(args)