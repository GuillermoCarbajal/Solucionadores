import numpy as np 
import pandas as pd

def agregar_datos_RUCAF_en_IAE_old(df_RUCAF, df_IAE):
    
    df_IAE_RUCAF_grouped = df_RUCAF.groupby('cedula').last()
    df_IAE_RUCAF_grouped = df_IAE_RUCAF_grouped.reset_index()

    for key in df_RUCAF.keys():
        if key!='cedula':
            df_IAE[f"RUCAF_{key}"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")[key])

    
    #df_IAE["RUCAF_prestador"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["prestador"])
    #df_IAE["RUCAF_pais"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["pais"])
    #df_IAE["RUCAF_departamento"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["departamento"])
    #df_IAE["RUCAF_localidad"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["localidad"])
    #df_IAE["RUCAF_cobertura"] = df_IAE["CEDULA"].map(df_IAE_RUCAF_grouped.set_index("cedula")["cobertura"])

    return df_IAE

def agregar_RUCAF_region(df_RUCAF):
    def get_region(departamento):
        Oeste = ['SAN JOSE', 'FLORES', 'FLORIDA', 'COLONIA', 'SORIANO','RIO NEGRO','DURAZNO']
        Norte = ['ARTIGAS', 'SALTO', 'PAYSANDU', 'RIVERA', 'TACUAREMBO']
        Este = ['MALDONADO', 'ROCHA', 'LAVALLEJA', 'TREINTA Y TRES', 'CERRO LARGO']
        Sur = ['CANELONES', 'MONTEVIDEO']
        
        region = None
        if departamento in Oeste:
            region = 'Oeste'
        elif departamento in Norte:
            region = 'Norte'
        elif departamento in Este:
            region='Este'
        elif departamento in Sur:
            region='Sur' 
        return region
    
    df_RUCAF['region_'] = df_RUCAF['departamento'].map(get_region)

    return df_RUCAF

def agregar_datos_RUCAF_en_IAE(intento, df_RUCAF_agrupada):
    cedula = intento['CEDULA']
    
    RUCAF_prestador = []
    RUCAF_pais = []
    RUCAF_departamento = []
    RUCAF_localidad = []
    RUCAF_cobertura = []
    RUCAF_tipo_prestador = []
    RUCAF_region = []
    total_prestadores = 0
    
    if cedula in df_RUCAF_agrupada.groups:

    #else:
        datos_RUCAF = df_RUCAF_agrupada.get_group(cedula)
        prestadores = datos_RUCAF['prestador']
        coberturas = datos_RUCAF['cobertura']
        departamentos = datos_RUCAF['departamento']
        regiones = datos_RUCAF['region_']
        localidades = datos_RUCAF['localidad']
        paises = datos_RUCAF['pais']
        tipo_prestadores_RUCAF = datos_RUCAF['tipo_prestador_RUCAF']
        total_prestadores = prestadores.shape[0]

        for i in range(total_prestadores):
            RUCAF_prestador.append(str(prestadores.iloc[i]))
            RUCAF_pais.append(str(paises.iloc[i]))
            RUCAF_departamento.append(str(departamentos.iloc[i]))
            RUCAF_region.append(str(regiones.iloc[i]))
            RUCAF_localidad.append(str(localidades.iloc[i]))
            RUCAF_cobertura.append(str(coberturas.iloc[i]))
            RUCAF_tipo_prestador.append(str(tipo_prestadores_RUCAF.iloc[i]))
        #print(f'La persona tiene {total_hijos} en total y tenía {cantidad_hijos} en la fecha del intento')
 
        edad_hijo_menor = np.nan
        # Se crea una variable booleana que indica si tiene asociado al menos un prestador público o no 
    RUCAF_PRESTADOR_PUBLICO_ = 'Pública' in RUCAF_tipo_prestador 
    RUCAF_PRESTADOR_PRIVADO_ = 'Privada' in RUCAF_tipo_prestador
    RUCAF_fonasa = 'Fonasa' in RUCAF_cobertura
    RUCAF_no_fonasa = 'No Fonasa' in RUCAF_cobertura    

    nuevos_campos = {'RUCAF_prestador':'|'.join(RUCAF_prestador), 
                     'RUCAF_tipo_prestador':'|'.join(RUCAF_tipo_prestador),
                     'RUCAF_pais':'|'.join(RUCAF_pais), 
                     'RUCAF_departamento':'|'.join(RUCAF_departamento),
                     'RUCAF_region_':'|'.join(RUCAF_region),
                     'RUCAF_localidad':'|'.join(RUCAF_localidad),
                     'RUCAF_cobertura':'|'.join(RUCAF_cobertura),
                     'RUCAF_total_prestadores':total_prestadores,
                     'RUCAF_PRESTADOR_PUBLICO_':RUCAF_PRESTADOR_PUBLICO_,
                     'RUCAF_PRESTADOR_PRIVADO_':RUCAF_PRESTADOR_PRIVADO_,
                     'RUCAF_fonasa_': RUCAF_fonasa,
                     'RUCAF_no_fonasa_': RUCAF_no_fonasa
                     }
    
    #df_result = pd.concat([intento, nuevos_campos])

    result = intento.copy()
    for k, v in nuevos_campos.items():
        result[k] = v

    return result