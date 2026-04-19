import numpy as np 
import pandas as pd

def agregar_datos_hijos_cuando_intento(intento, df_IAE, df_IAE_CNV_agrupada):
    cedula = intento['CEDULA']
    if cedula not in df_IAE_CNV_agrupada.groups:
        cantidad_hijos = 0
        edad_hijo_mayor = None
        edad_hijo_menor = None
    else:
        datos_cnv = df_IAE_CNV_agrupada.get_group(cedula)
        fecha_intento = intento['FECHA IAE']
        fechas_nacimientos = datos_cnv['anio_mes_nacimiento_hijo']
        total_hijos = fechas_nacimientos.shape[0]
        cantidad_hijos = np.sum(fechas_nacimientos < fecha_intento)
        print(f'La persona tiene {total_hijos} en total y tenía {cantidad_hijos} en la fecha del intento')
        if cantidad_hijos>0:
            nacimientos_antes_intento = fechas_nacimientos[fechas_nacimientos<fecha_intento]
            edad_hijo_mayor = (fecha_intento - nacimientos_antes_intento.min())
            edad_hijo_mayor = edad_hijo_mayor.days/365.25
            edad_hijo_menor = (fecha_intento - nacimientos_antes_intento.max())
            edad_hijo_menor = edad_hijo_menor.days/365.25
        else:
            edad_hijo_mayor = np.nan
            edad_hijo_menor = np.nan
        
    #df_IAE[["cantidad_hijos", "edad_hijo_mayor", "edad_hijo_menor"]] = pd.Series({'cant_hijos':cantidad_hijos, 'edad_hijo_mayor': edad_hijo_mayor, 'edad_hijo_menor':edad_hijo_menor})
    nuevos_campos = {'CNV_cant_hijos_cuando_IAE':cantidad_hijos, 'CNV_edad_hijo_mayor_cuando_IAE': edad_hijo_mayor, 'CNV_edad_hijo_menor_cuando_IAE':edad_hijo_menor}
    
    #df_result = pd.concat([intento, nuevos_campos])

    result = intento.copy()
    for k, v in nuevos_campos.items():
        result[k] = v

    return result

def agregar_datos_CNV_cuando_intento(intento, df_IAE, df_IAE_CNV_agrupada):
    cedula = intento['CEDULA']
    campos_CNV = [ 'cedula', 'anio_mes_nacimiento_hijo', 'nro_rese', 'estado_civil', 'pais_nac', 'pais_residencia', 
                'mayor_nivel_estudio', 'sexo', 'peso', 'semanas_gestacion', 'orden', 'certdatospartocodocurrencia', 
                'apgar1', 'apgar2', 'tipo_gesta', 'numero_embarazo_anteriores', 'semana_embarazo_primer_consulta', 
                'total_consultas', 'mes_parto', 'lugar_parto', 'lugar_depto', 'tipo_certificador', 'tipo_parto',
                'tipo_cesarea', 'forceps', 'vaccum', 'otra_maniobra']
    nuevos_campos={}
    if cedula not in df_IAE_CNV_agrupada.groups:
        for campo in campos_CNV:
            nuevos_campos['CNV_ultimo_cuando_IAE_' + campo ] = None
    else:
        datos_cnv = df_IAE_CNV_agrupada.get_group(cedula)
        fecha_intento = intento['FECHA IAE']

        fechas_nacimientos = datos_cnv['anio_mes_nacimiento_hijo']
        cantidad_hijos = np.sum(fechas_nacimientos < fecha_intento)
        total_hijos = fechas_nacimientos.shape[0]
        print(f'La persona tiene {total_hijos} en total y tenía {cantidad_hijos} en la fecha del intento')
        for campo in campos_CNV:
            
            if cantidad_hijos>0:
                #print(campo)
                #print(datos_cnv.keys())
                datos_campo = datos_cnv[campo]
                #print(datos_campo)
                datos_campo_de_interes = datos_campo[fechas_nacimientos<fecha_intento]
                #print(datos_campo_de_interes.iloc[-1])
                nuevos_campos['CNV_ultimo_cuando_IAE_' + campo ] = datos_campo_de_interes.iloc[-1]
            else:
                nuevos_campos['CNV_ultimo_cuando_IAE_' + campo] = None
        
    #df_result = pd.concat([intento, pd.Series(nuevos_campos)])

    result = intento.copy()
    for k, v in nuevos_campos.items():
        result[k] = v
    
    return result