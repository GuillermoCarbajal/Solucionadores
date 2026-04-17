import pandas as pd
import numpy as np
from .utils import obtener_palabras_en_campo_que_contienen_substr

def filtrar_IAES_por_fecha_registro(df_IAE, fecha_inicial, fecha_fin):
    '''
    Ejemplo:
    fecha_inicial = '2023-07-01'
    fecha_fin = '2023-08-01'
    '''
    return df_IAE[(df_IAE['REGISTRO'] >= fecha_inicial) & (df_IAE['REGISTRO'] < fecha_fin)]


def etiquetar_prestador(institucion):
    ''' 
    Si dice ASSE, POLICIAL, FUERZAS ARMADAS o CLINICAS etiqueto como Pública, si no Privada
    '''
    if 'ASSE' in institucion or 'POLICIAL' in institucion or 'FUERZAS ARMADAS' in institucion or 'CLINICAS' in institucion:
        return 'Pública'
    else:
        return 'Privada'

def tipo_prestador_IAE(institucion):
    ''' 
    Etiqueta los prestadores asociados a un IAE. Puede haber varios prestadores asociados al intento. Se separan por el símbolo "|".
    Ej: Si hay varios prestadores podría devolver: "Pública | Privada | Pública" o "Privada | Pública"
    '''
    if institucion=='No indicado':
        return institucion
    elif institucion == 'NaN' or pd.isna(institucion):
        return 'nan'
    elif '|' in institucion:
        instituciones =  institucion.split('|')
        etiqueta = [etiquetar_prestador(inst) for inst in instituciones]
        etiqueta = '|'.join(etiqueta)
        #print(institucion, etiqueta)
        return etiqueta
    else:
        return etiquetar_prestador(institucion)

def agregar_tipo_prestador_IAE(df_IAE, dataset):
    # Se crea una nueva columna con los tipos de prestador
    if dataset==2:
        df_IAE['Tipo_prestador_IAE_'] = df_IAE['PRESTADOR'].copy()
        df_IAE['Tipo_prestador_IAE_'] = df_IAE['PRESTADOR'].apply(lambda x: tipo_prestador_IAE(x) if pd.notnull(x) else np.nan)
    else:
        df_IAE['Tipo_prestador_IAE_'] = df_IAE['PRESTADOR RECODIFICADO'].copy()
        df_IAE['Tipo_prestador_IAE_'] = df_IAE['PRESTADOR RECODIFICADO'].apply(lambda x: tipo_prestador_IAE(x) if pd.notnull(x) else np.nan)

    # Se crea una variable booleana que indica si tiene asociado al menos un prestador público o no 
    df_IAE['PRESTADOR_PUBLICO_'] = df_IAE['Tipo_prestador_IAE_'].apply(lambda x: 'Pública' in x if pd.notnull(x) else np.nan)
    df_IAE['PRESTADOR_PRIVADO_'] = df_IAE['Tipo_prestador_IAE_'].apply(lambda x: 'Privada' in x if pd.notnull(x) else np.nan) 

    return df_IAE

def guardar_prestadores(df_IAE, dataset):
    # Guardo todos los prestadores que aparecen en la tabla en una archivo para que quede. Quizás haya repetidos con pequeñas diferencias.
    if dataset==2:
        nombre_prestadores = 'prestadores_2a_entrega.csv'
        campo_prestador = 'PRESTADOR'
    else:
        nombre_prestadores = 'prestadores.csv'
        campo_prestador = 'PRESTADOR RECODIFICADO'

    with open(nombre_prestadores,'w') as f:
        for i, prestador in enumerate(df_IAE[campo_prestador].unique()):
            print(i, prestador)
            f.write(f'{prestador}, {tipo_prestador_IAE(prestador)} \n')
    f.close()


def acondicionar_atributo_metodo(df_IAE):
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("OBSTRUCCI�N", 'OBSTRUCCIÓN')
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("CA�DA","CAÍDA")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("RESPIRACI�N", "RESPIRACIÓN")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("INGESTI�N", "INGESTIÓN")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("INHALACI�N", "INHALACIÓN")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("V�AS", "VÍAS")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("EXPOSICI�N", "EXPOSICIÓN")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("G�STRICOS", "GÁSTRICOS")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("EL�CTRICA", "ELÉCTRICA")
    df_IAE['METODO'] = df_IAE['METODO'].str.replace("FR�O", "FRÍO")
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','AHORCAMIENTO')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Ahorcamiento o asfixia')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','ARMAS DE FUEGO')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Armas de fuego')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','Arma de fuego')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Armas de fuego')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','AUTOINFLIGIDA')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Lesiones autoinfligidas')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','FACTORES NO ESPECIFICADOS')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Otros')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','NO ESPECIFICADA DE LA RESPIRACIÓN')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Ahorcamiento o asfixia')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','OBSTRUCCIÓN DE LAS VÍAS RESPIRATORIAS')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Ahorcamiento o asfixia')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','AHOGAMIENTO Y SUMERSI')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Ahogamiento')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','FUEGO')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Fuego')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','CAÍDA')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Caída')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','ENVENENAMIENTO AUTOINFLIGIDO INTENCIONALMENTE')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Envenenamiento autoinfligido intencionalmente')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','ENVENENAMIENTO ACCIDENTAL')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Envenenamiento accidental')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','ENVENENAMIENTO')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Envenenamiento')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','EVENTO NO ESPECIFICADO')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Otros')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','INHALACIÓN DE CONTENIDOS GÁSTRICOS')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Inhalación de contenidos gástricos')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','CORRIENTE ELÉCTRICA')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Corriente eléctrica')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, 'METODO','EXPOSICIÓN AL FRÍO NATURAL')
    df_IAE['METODO'] = df_IAE['METODO'].replace(palabras_a_reemplazar,'Frío Natural')

    df_IAE["METODO"] = df_IAE["METODO"].replace("otros", "Otros")

    return df_IAE

def agregar_categoria_metodo(df_IAE, nombre='METODO_'):
    
    categorias = ['Ingesta de Medicamentos', 'Ahorcamiento o asfixia','Lesiones autoinfligidas', 'Armas de fuego', 'Caída']
    df_IAE[nombre] = df_IAE['METODO'].where(df_IAE['METODO'].isin(categorias), 'Otros')
    
    return df_IAE

def acondicionar_IAE_PREVIO(df_IAE):

    df_IAE['IAE PREVIO'] = df_IAE['IAE PREVIO'].str.strip() # Elimino espacios antes y después de los string
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE,'IAE PREVIO','Tratamiento')
    df_IAE['IAE PREVIO'] = df_IAE['IAE PREVIO'].replace(palabras_a_reemplazar,'Tratamiento')
    df_IAE['IAE PREVIO'] = df_IAE['IAE PREVIO'].replace('Sin dato','NO SE INDICA')
    df_IAE['IAE PREVIO'] = df_IAE['IAE PREVIO'].replace('','NO SE INDICA')
    df_IAE['IAE PREVIO'] = df_IAE['IAE PREVIO'].fillna("NO SE INDICA")
    
    return df_IAE

def calcular_y_agregar_campo_edad(df_IAE, campo_fecha, campo_nacimiento):
    df_IAE[campo_fecha] = pd.to_datetime(df_IAE[campo_fecha], errors='coerce')
    df_IAE[campo_nacimiento] = pd.to_datetime(df_IAE[campo_nacimiento], errors='coerce')
    df_IAE['EDAD_'] = df_IAE[campo_fecha].dt.year - df_IAE[campo_nacimiento].dt.year

    return df_IAE


def acondicionar_campo_DECISION(df_IAE, campo_decision):
    df_IAE[campo_decision] = df_IAE[campo_decision].str.strip() # saco espacios adelante y atras
    #df_IAE[campo_decision].value_counts()
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'RESUELTO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'RESUELTO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'NO CUMPLE')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'NO CUMPLE PROTOCOLO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'SUICIDIO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'SUICIDIO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'INTERNAD')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'INTERNADO')
    #df_IAE[campo_decision] = df_IAE[campo_decision].replace('INTERNADO ','INTERNADO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('PENDIENTE INTERNADO','INTERNADO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'PENDIENTE RESPUESTA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'PENDIENTE RESPUESTA')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'PENDEINTE RESPUESTA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'PENDIENTE RESPUESTA')

    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN RESPUESTA DEFINITIVA','PENDIENTE RESPUESTA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN RESPUESTA','PENDIENTE RESPUESTA')
    #df_IAE[campo_decision].unique()
    return df_IAE

def agregar_campo_DECISION(df_IAE, campo_decision, nuevo_nombre):
    # se crea una nueva columna DECISION_ con 6 categorías.
    df_IAE[nuevo_nombre] = df_IAE[campo_decision].copy()

    categorias = ['RESUELTO', 'NO CUMPLE PROTOCOLO','PENDIENTE RESPUESTA','INTERNADO', 'SUICIDIO']
    df_IAE[nuevo_nombre] = df_IAE[nuevo_nombre].where(df_IAE[nuevo_nombre].isin(categorias), 'OTRA DECISION')

    return df_IAE

def agregar_si_intentos_en_CDE(df_IAE, df_IAE_CDE, nombre_nuevo_campo='DEFUNCION_'):
    df_IAE[nombre_nuevo_campo] = df_IAE["CEDULA"].isin(df_IAE_CDE["cedula"]).astype(int)
    cantidad_intentos_fallecidos = np.sum(df_IAE[nombre_nuevo_campo])
    print(f'Se agrego el campo {nombre_nuevo_campo} que vale 1 si la persona falleció (no necesariamente suicidio)')
    print(f'{cantidad_intentos_fallecidos} de los {df_IAE.shape[0]} intentos están asociados a personas fallecidas') 

    return df_IAE


def agregar_datos_CDE_en_IAE(df_IAE, df_IAE_CDE, dataset):
    
    df_IAE["FECHA_DEFUNCION"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["fecha_defuncion"])
    df_IAE["CAUSA_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["causa_basica_muerte_valor"])
    df_IAE["DPTO_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["departamento_ocurrencia"])
    df_IAE["EDAD_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["edad_fallecimiento_digitada"])
    df_IAE["GRUPO_EDAD_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["grupo edades_"])
    df_IAE["CAT_SUI_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["CAT_SUI_"])
    df_IAE["CAT_MCEXSUI_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["CAT_MCEXSUI_"])
    
    if dataset==2:
        df_IAE["MOTIVO_EXTERNO_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["motivo_externo"])
        df_IAE["MOTIVO_EXT_SUI_"] = df_IAE["MOTIVO_EXTERNO_"]=='SUICIDIO'
        df_IAE["ES_MOTIVO_EXTERNO_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["es_motivo_externo"])

    return df_IAE

def agregar_dias_IAE_a_muerte(df_IAE, nombre_nuevo_campo='DIAS_IAE_MUERTE_'):
    
    # días que transcurren desde el IAE hasta la muerte. Si la persona no murió se pone nan
    df_IAE[nombre_nuevo_campo] = (df_IAE["FECHA_DEFUNCION"] - df_IAE["FECHA IAE"])

    return df_IAE