import pandas as pd
import numpy as np
from .utils import obtener_palabras_en_campo_que_contienen_substr, convertir_enteros_a_fecha

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
    if 'ASSE' in institucion or 'POLICIAL' in institucion or 'FUERZAS ARMADAS' in institucion \
        or 'FF.AA' in institucion or 'CLINICAS' in institucion:
        
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

def agregar_tipo_prestador_IAE(df_IAE, campo_prestador='PRESTADOR RECODIFICADO', nombre_nuevo_campo='Tipo_prestador_IAE_'):
    # Se crea una nueva columna con los tipos de prestador
    #if dataset==2:
    #    df_IAE[nombre_nuevo_campo] = df_IAE['PRESTADOR'].copy()
    #    df_IAE[nombre_nuevo_campo] = df_IAE['PRESTADOR'].apply(lambda x: tipo_prestador_IAE(x) if pd.notnull(x) else np.nan)
    #else:
    #    df_IAE[nombre_nuevo_campo] = df_IAE['PRESTADOR RECODIFICADO'].copy()
    #    df_IAE[nombre_nuevo_campo] = df_IAE['PRESTADOR RECODIFICADO'].apply(lambda x: tipo_prestador_IAE(x) if pd.notnull(x) else np.nan)

    df_IAE[nombre_nuevo_campo] = df_IAE[campo_prestador].copy()
    df_IAE[nombre_nuevo_campo] = df_IAE[campo_prestador].apply(lambda x: tipo_prestador_IAE(x) if pd.notnull(x) else np.nan)

    # Se crea una variable booleana que indica si tiene asociado al menos un prestador público o no 
    df_IAE['PRESTADOR_PUBLICO_'] = df_IAE[nombre_nuevo_campo].apply(lambda x: 'Pública' in x if pd.notnull(x) else np.nan)
    df_IAE['PRESTADOR_PRIVADO_'] = df_IAE[nombre_nuevo_campo].apply(lambda x: 'Privada' in x if pd.notnull(x) else np.nan) 

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
    df_IAE.loc[df_IAE['EDAD_'] > 150 ,'EDAD_'] = np.nan
    df_IAE.loc[df_IAE['EDAD_'] < 0,'EDAD_' ] = np.nan

    return df_IAE


def acondicionar_campo_DECISION(df_IAE, campo_decision):
    df_IAE[campo_decision] = df_IAE[campo_decision].str.strip() # saco espacios adelante y atras
    #df_IAE[campo_decision].value_counts()
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'RESUELTO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'RESUELTO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'RESUELTO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'SI')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'NO CUMPLE')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'NO CUMPLE PROTOCOLO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'SUICIDIO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'SUICIDIO')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'INTERNAD')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'INTERNADO')
    #df_IAE[campo_decision] = df_IAE[campo_decision].replace('INTERNADO ','INTERNADO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('PENDIENTE INTERNADO','INTERNADO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('INTERNADO','PENDIENTE RESPUESTA')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'PENDIENTE RESPUESTA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'PENDIENTE RESPUESTA')
    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'PENDEINTE RESPUESTA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'PENDIENTE RESPUESTA')

    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN RESPUESTA DEFINITIVA','PENDIENTE RESPUESTA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN RESPUESTA','PENDIENTE RESPUESTA')
    #df_IAE[campo_decision].unique()
    return df_IAE

def acondicionar_campo_agendo_consulta_en_7dias(df_IAE, campo_decision):
    
    # llamo al campo decision porque así se llamaba en la primera entrega (reutilizo codigo)
    df_IAE = acondicionar_campo_DECISION(df_IAE, campo_decision)
    
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('DESCARTDO','DESCARTADO',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('DECARTADA','DESCARTADA',regex=True)

    

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'DESCARTAD')
    #print(palabras_a_reemplazar)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'DESCARTADA_POR_RASTREADOR')

    palabras_a_reemplazar = obtener_palabras_en_campo_que_contienen_substr(df_IAE, campo_decision,'SIN COBERTURA')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace(palabras_a_reemplazar,'SIN COBERTURA ASISTENCIAL')

    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SEG PARTICULAR','SEGUIMIENTO PARTICULAR')
    df_IAE['DESCARTADA_POR_RASTREADOR'] = df_IAE[campo_decision]  == 'DESCARTADA_POR_RASTREADOR'

    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN COBERTURA ASISTENCIAL','NO',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN DATO ASISTENCIAL','NO',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN REGISTRO','NO',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SIN RESPUESTA','NO',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('NO CUMPLE PROTOCOLO','NO')
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('PRIVADO DE LIBERTAD','NO')

    df_IAE[campo_decision] = df_IAE[campo_decision].replace('RESUELTO','SI',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('SEGUIMIENTO PARTICULAR','SI',regex=True)
    df_IAE[campo_decision] = df_IAE[campo_decision].replace('RESUELTO','SI',regex=True)
    
    
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

def agregar_si_es_IAE(df_IAE,  nombre_nuevo_campo='es_IAE_'):
    df_IAE[nombre_nuevo_campo] = ~df_IAE["FECHA IAE"].isnull()
    cantidad_intentos = np.sum(df_IAE[nombre_nuevo_campo])
    print(f'{cantidad_intentos} de las {df_IAE.shape[0]} filas de la base IAE son intentos') 

    return df_IAE

def agregar_si_tiene_fecha_registro(df_IAE, nombre_nuevo_campo='tiene_registro_'):
    df_IAE[nombre_nuevo_campo] = ~df_IAE["REGISTRO"].isnull()
    cantidad_registros = np.sum(df_IAE[nombre_nuevo_campo])
    print(f'{cantidad_registros} de las {df_IAE.shape[0]} filas de la base IAE tienen fecha de registro') 

    return df_IAE

def corregir_fechas_enteras_prestadores(df_IAE):
    campos_a_corregir = ['FECHA SEGUIMIENTO', 'FECHA LLAMADA PRESTADOR', 'AGENDO CONSULTA ESM 7DIAS SI/NO/INTERNADO',
                        'FECHA DE CONSULTA', 'CONCURRIO', 'SE LLAMO A USUARIO Y/O REFERENTE',
                        'AGENDO NUEVA CONSULTA', 'FECHA DE NUEVA CONSULTA', 'INTERNACION',
                        'FECHA ALTA', 'FECHA LLAMADA PRESTADOR.1', 'MSP', 'OBSERVACIONES',
                        'UNNAMED: 26', 'FECHA DE LLAMADA AL PRESTADOR', 'MOTIVO']
    campos_corregidos =  [   campo + '_' for campo in campos_a_corregir]


    #print(df_IAE[campos_a_corregir].dtypes)

    #print(df_IAE[campos_a_corregir].head())

    #print(type(df_IAE[campos_a_corregir].iloc[0]))
    
    df_IAE[campos_corregidos] = df_IAE[campos_a_corregir].apply(convertir_enteros_a_fecha)

    for campo, campo_corregido in zip(campos_a_corregir, campos_corregidos):
        cambios = ~(df_IAE[campo].eq(df_IAE[campo_corregido]) |
                   (df_IAE[campo].isna() & df_IAE[campo_corregido].isna()))
        cantidad_cambio = np.sum(cambios)
        print(f'En el campo {campo} se corrigieron {cantidad_cambio} fechas enteras')

    return df_IAE

def acondicionar_campos_prestadores(df_IAE):
    campos_a_corregir = ['FECHA SEGUIMIENTO', 'FECHA LLAMADA PRESTADOR', 'AGENDO CONSULTA ESM 7DIAS SI/NO/INTERNADO',
                        'FECHA DE CONSULTA', 'CONCURRIO', 'SE LLAMO A USUARIO Y/O REFERENTE',
                        'AGENDO NUEVA CONSULTA', 'FECHA DE NUEVA CONSULTA', 'INTERNACION',
                        'FECHA ALTA', 'FECHA LLAMADA PRESTADOR.1', 'MSP', 'OBSERVACIONES',
                        'UNNAMED: 26', 'FECHA DE LLAMADA AL PRESTADOR', 'MOTIVO']
    campos_a_corregir =  [   campo + '_' for campo in campos_a_corregir]

    for campo in campos_a_corregir:
        # saco espacios adelante y atras y paso a mayúsculas
        df_IAE[campo] = df_IAE[campo].where(df_IAE[campo].isna(),df_IAE[campo].astype(str).str.strip().str.upper())
                                            
    #df_IAE[campo_decision].unique()
    return df_IAE

def acondicionar_CONCURRIO(df_IAE, nombre_campo='CONCURRIO_', nombre_nuevo_campo='CONCURRIO_binario'):
    
    es_fecha = pd.to_datetime(df_IAE[nombre_campo], errors="coerce").notna()
    df_IAE[nombre_nuevo_campo] = df_IAE[nombre_campo].where(~es_fecha, "SI")
    df_IAE['NO_CONCURRIO_CONSULTA_'] = df_IAE['CONCURRIO_']=='NO'

    return df_IAE

def agregar_fecha_IAE_siguiente(df_IAE):
    # Guardar el orden original
    df_IAE["_orden_original"] = np.arange(len(df_IAE))
    df_IAE = df_IAE.sort_values(["CEDULA", "FECHA IAE"])

    df_IAE["FECHA IAE SIGUIENTE"] = (
        df_IAE.groupby("CEDULA")["FECHA IAE"]
          .shift(-1)
    )

    # Restaurar el orden original
    df_IAE = (
        df_IAE.sort_values("_orden_original")
          .drop(columns="_orden_original")
          .reset_index(drop=True)
    )

    return df_IAE

def agregar_info_intentos_previos(df_IAE):
    # Guardar el orden original
    df_IAE["_orden_original"] = np.arange(len(df_IAE))
    df_IAE = df_IAE.sort_values(["CEDULA", "FECHA IAE"])

    df_IAE["n_intento_"] = (
        df_IAE["FECHA IAE"]
        .notna()
        .groupby(df_IAE["CEDULA"])
        .cumsum())



    df_IAE['IAE_PREVIO_CORREGIDO_'] = df_IAE['IAE PREVIO'].copy()
    indices_no_se_indica_a_cambiar = (df_IAE["IAE_PREVIO_CORREGIDO_"] == "NO SE INDICA") & (df_IAE["n_intento_"] > 1)
    df_IAE.loc[indices_no_se_indica_a_cambiar, "IAE_PREVIO_CORREGIDO_"] = "SI"
    print(f'Se corrigieron {np.sum(indices_no_se_indica_a_cambiar)} intentos que decían NO SE INDICA en campo IAE PREVIO')

    indices_no_a_cambiar = (df_IAE["IAE_PREVIO_CORREGIDO_"] == "NO") & (df_IAE["n_intento_"] > 1)
    df_IAE.loc[indices_no_a_cambiar, "IAE_PREVIO_CORREGIDO_"] = "SI" 
    print(f'Se corrigieron {np.sum(indices_no_a_cambiar)} intentos que decían NO en campo IAE PREVIO')
    

    df_IAE["total_intentos_"] = (
    df_IAE.groupby("CEDULA")["FECHA IAE"]
          .transform("count"))

    campos = ['METODO'] # FECHA IAE

    for campo in campos:
        df_IAE[f"{campo}_IAE_PREVIO_"] = (
            df_IAE.groupby("CEDULA")[campo]
            .shift(1)
        )
        df_IAE[f"{campo}_IAE_PREVIO_2_"] = (
            df_IAE.groupby("CEDULA")[campo]
            .shift(2)
        )
    
    
    df_IAE["DIAS_DESDE_IAE_PREVIO_"] = (
        df_IAE.groupby("CEDULA")["FECHA IAE"]
        .diff()
        .dt.days)    
    

    df_IAE["PROMEDIO_DIAS_ENTRE_IAES_"] = (
        df_IAE.groupby("CEDULA")["DIAS_DESDE_IAE_PREVIO_"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True))

    
    df_IAE["STD_DIAS_ENTRE_IAES_"] = (
        df_IAE.groupby("CEDULA")["DIAS_DESDE_IAE_PREVIO_"]
        .expanding()
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    
    # Restaurar el orden original
    df_IAE = (
        df_IAE.sort_values("_orden_original")
        .drop(columns="_orden_original")
        .reset_index(drop=True)
    )



    return df_IAE


def agregar_tiempo_reincidencia(df_IAE):
    tr1 = (df_IAE['FECHA IAE SIGUIENTE'] - df_IAE['FECHA IAE']).dt.days
    df_IAE['tiempo_reintento_']=tr1
    df_IAE['reint_2dias_'] = df_IAE['tiempo_reintento_'] < 2 #pd.Timedelta(days=2)
    df_IAE['reint_5dias_'] = df_IAE['tiempo_reintento_'] < 5 #pd.Timedelta(days=5)
    df_IAE['reint_10dias_'] = df_IAE['tiempo_reintento_'] < 10 #pd.Timedelta(days=10)
    df_IAE['reint_30dias_'] = df_IAE['tiempo_reintento_'] < 30 #pd.Timedelta(days=30)
    df_IAE['reint_60dias_'] = df_IAE['tiempo_reintento_'] < 60 #pd.Timedelta(days=60)
    df_IAE['reint_90dias_'] = df_IAE['tiempo_reintento_'] < 90 #pd.Timedelta(days=90)
    df_IAE['reintento_'] = ~pd.isnull(df_IAE['tiempo_reintento_'])
    
    tr2 = df_IAE["DIAS_IAE_MUERTE_"].dt.days.where(df_IAE["CAT_SUI_"] == 1)
    df_IAE['tiempo_suicidio_'] = tr2
    df_IAE['suicidio_2dias_'] = df_IAE['tiempo_suicidio_'] < 2 #pd.Timedelta(days=2)
    df_IAE['suicidio_5dias_'] = df_IAE['tiempo_suicidio_'] < 5 #pd.Timedelta(days=5)
    df_IAE['suicidio_10dias_'] = df_IAE['tiempo_suicidio_'] < 10 #pd.Timedelta(days=10)
    df_IAE['suicidio_30dias_'] = df_IAE['tiempo_suicidio_'] < 30 #pd.Timedelta(days=30)
    df_IAE['suicidio_60dias_'] = df_IAE['tiempo_suicidio_'] < 60 #pd.Timedelta(days=60)
    df_IAE['suicidio_90dias_'] = df_IAE['tiempo_suicidio_'] < 90 #pd.Timedelta(days=90)
    df_IAE['suicidio_'] = ~pd.isnull(df_IAE['tiempo_suicidio_'])

    tr3 = df_IAE["DIAS_IAE_MUERTE_"].dt.days
    df_IAE['tiempo_muerte_'] = tr3
    df_IAE['muerte_2dias_'] = df_IAE['tiempo_muerte_'] < 2 #pd.Timedelta(days=2)
    df_IAE['muerte_5dias_'] = df_IAE['tiempo_muerte_'] < 5 #pd.Timedelta(days=5)
    df_IAE['muerte_10dias_'] = df_IAE['tiempo_muerte_'] < 10 #pd.Timedelta(days=10)
    df_IAE['muerte_30dias_'] = df_IAE['tiempo_muerte_'] < 30 #pd.Timedelta(days=30)
    df_IAE['muerte_60dias_'] = df_IAE['tiempo_muerte_'] < 60 #pd.Timedelta(days=60)
    df_IAE['muerte_90dias_'] = df_IAE['tiempo_muerte_'] < 90 #pd.Timedelta(days=90)
    df_IAE['muerte_'] = ~pd.isnull(df_IAE['tiempo_muerte_'])
    
    
    df_IAE['tiempo_reintento_suicidio_']  = tr2.where(pd.isnull(df_IAE['FECHA IAE SIGUIENTE']), other=tr1)
    df_IAE['reint_suicidio_2dias_'] = df_IAE['tiempo_reintento_suicidio_'] < 2 #pd.Timedelta(days=2)
    df_IAE['reint_suicidio_5dias_'] = df_IAE['tiempo_reintento_suicidio_'] < 5 #pd.Timedelta(days=5)
    df_IAE['reint_suicidio_10dias_'] = df_IAE['tiempo_reintento_suicidio_'] < 10 #pd.Timedelta(days=10)
    df_IAE['reint_suicidio_30dias_'] = df_IAE['tiempo_reintento_suicidio_'] < 30 #pd.Timedelta(days=30)
    df_IAE['reint_suicidio_60dias_'] = df_IAE['tiempo_reintento_suicidio_'] < 60 #pd.Timedelta(days=60)
    df_IAE['reint_suicidio_90dias_'] = df_IAE['tiempo_reintento_suicidio_'] < 90 #pd.Timedelta(days=90)
    df_IAE['reintento_suicidio_'] = ~pd.isnull(df_IAE['tiempo_reintento_suicidio_']) 

    df_IAE['tiempo_reintento_muerte_']  = tr3.where(pd.isnull(df_IAE['FECHA IAE SIGUIENTE']), other=tr1)
    df_IAE['reint_muerte_2dias_'] = df_IAE['tiempo_reintento_muerte_'] < 2 #pd.Timedelta(days=2)
    df_IAE['reint_muerte_5dias_'] = df_IAE['tiempo_reintento_muerte_'] < 5 #pd.Timedelta(days=5)
    df_IAE['reint_muerte_10dias_'] = df_IAE['tiempo_reintento_muerte_'] < 10 #pd.Timedelta(days=10)
    df_IAE['reint_muerte_30dias_'] = df_IAE['tiempo_reintento_muerte_'] < 30 #pd.Timedelta(days=30)
    df_IAE['reint_muerte_60dias_'] = df_IAE['tiempo_reintento_muerte_'] < 60 #pd.Timedelta(days=60)
    df_IAE['reint_muerte_90dias_'] = df_IAE['tiempo_reintento_muerte_'] < 90 #pd.Timedelta(days=90)
    df_IAE['reintento_muerte_'] = ~pd.isnull(df_IAE['tiempo_reintento_muerte_']) 
  
    return df_IAE