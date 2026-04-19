import numpy as np 
import pandas as pd



def agregar_datos_EH_cuando_intento(intento, df_IAE, df_IAE_EH_agrupada):
    cedula = intento['CEDULA']
    campos_EH = ['cedula', 'Institución', 'Sector', 'Fecha ingreso', 'Fecha Egreso',
                  'Diagnostico principal según CIE-10', 'Causa externa según CIE-10']
    
    nuevos_campos = {}
    if cedula not in df_IAE_EH_agrupada.groups:
        for campo in campos_EH:
            nuevos_campos['EH_cercano_a_IAE_' + campo ] = None
    else:
        datos_cnv = df_IAE_EH_agrupada.get_group(cedula).reset_index()
        fecha_intento = intento['FECHA IAE']

        fechas_ingreso = datos_cnv['Fecha ingreso']
        cantidad_ingresos = fechas_ingreso.shape[0]
        diferencias = fechas_ingreso - fecha_intento

        tol_dias = 3
        if not pd.isnull(diferencias).all():
            indice_mas_cercana = np.argmin(np.abs(diferencias))
            fecha_mas_cercana = fechas_ingreso.iloc[indice_mas_cercana]
            dias_de_diferencia = diferencias[indice_mas_cercana]
            if np.abs(dias_de_diferencia.days) < tol_dias:
                print(f'La persona tuvo un intento el {fecha_intento}, tiene {cantidad_ingresos} ingresos, estos son {fechas_ingreso} ')
                print(f'El ingreso más cercano es {fecha_mas_cercana} ')
                print(f'Diferencia en días: {diferencias}, la menor es: {dias_de_diferencia}')
                datos_ingreso_mas_cercano = datos_cnv.iloc[indice_mas_cercana]
                for campo in campos_EH:
                    
                    #print(campo)
                    #print(datos_cnv.keys())
                    datos_campo = datos_ingreso_mas_cercano[campo]
                    
                    #print(datos_campo_de_interes.iloc[-1])
                    nuevos_campos['EH_cercano_a_IAE_' + campo ] = datos_campo
            else:
                print(f'Se descarto el egreso: {dias_de_diferencia} con el IAE \n')
                for campo in campos_EH:
                    nuevos_campos['EH_cercano_a_IAE_' + campo ] = None
        else:
            for campo in campos_EH:
                nuevos_campos['EH_cercano_a_IAE_' + campo ] = None
          
        
    #result = pd.concat([intento, pd.Series(nuevos_campos)])
    
    result = intento.copy()
    for k, v in nuevos_campos.items():
        result[k] = v

    return result