import numpy as np 
import pandas as pd



def agregar_datos_EH_cuando_intento(intento, df_IAE_EH_agrupada):

    cedula = intento['CEDULA']

    campos_EH = [
        'cedula', 'Institución', 'Sector', 'Fecha ingreso', 'Fecha Egreso',
        'Diagnostico principal según CIE-10', 'Causa externa según CIE-10',
        'EH_intento_suicidio', 'EH_ideacion_suicida',
        'EH_antecedente_autolesion', 'EH_vulnerabilidad_laboral',
        'EH_vulnerabilidad_economica', 'EH_conflicto_soporte_familiar',
        'EH_historia_trauma_infantil', 'EH_trastorno_depresivo',
        'EH_trastorno_bipolar', 'EH_consumo_sustancias',
        'EH_trastorno_personalidad', 'EH_ansiedad_y_estres',
        'EH_enfermedad_oncologica', 'dias_internacion_'
    ]

    nuevos_campos = {}

    # Inicializar los campos nuevos
    for campo in campos_EH:
        if campo.startswith('EH_'):
            nuevos_campos['EH_antecedente_' + campo] = None
        else:
            nuevos_campos['EH_cercano_a_IAE_' + campo] = None

    if cedula in df_IAE_EH_agrupada.groups:

        datos_eh = df_IAE_EH_agrupada.get_group(cedula).reset_index()

        datos_eh['Fecha Egreso'] = pd.to_datetime(
            datos_eh['Fecha Egreso'],
            errors='coerce'
        )

        fecha_intento = pd.to_datetime(
            intento['FECHA IAE'],
            errors='coerce'
        )

        if not pd.isnull(fecha_intento):

            eh_anteriores = datos_eh[
                datos_eh['Fecha Egreso'].notna() &
                (datos_eh['Fecha Egreso'] < fecha_intento)
            ]

            if not eh_anteriores.empty:

                # Último EH anterior
                indice_ultimo_egreso = eh_anteriores['Fecha Egreso'].idxmax()
                datos_ultimo_eh = datos_eh.loc[indice_ultimo_egreso]

                for campo in campos_EH:

                    if campo.startswith('EH_'):
                        nuevos_campos['EH_antecedente_' + campo] = int(
                            eh_anteriores[campo].eq(1).any()
                        )
                    else:
                        nuevos_campos['EH_cercano_a_IAE_' + campo] = (
                            datos_ultimo_eh[campo]
                        )

    # Mantener primero EXACTAMENTE las columnas de intento
    # y agregar los nuevos campos al final
    nuevos_campos = pd.Series(nuevos_campos)

    result = pd.concat([
        intento,
        nuevos_campos
    ])

    return result





def generar_indicadores_EH(df_EH):
    # Definimos las columnas de diagnóstico a analizar
    columnas_diagnostico = [
        'Diagnostico principal según CIE-10', 
        'Causa externa según CIE-10', 
        'Diagnóstico complementario según CIE-10) 3',
        'Diagnóstico complementario según CIE-10) 4', 
        'Diagnóstico complementario seg7n CIE-10) 5',
        'Diagnóstico complementario según CIE-10) 6'
    ]

    # Aseguramos que todas las columnas sean tratadas como strings limpios de espacios
    # Esto se hace una sola vez para todo el DataFrame
    df_temp = df_EH[columnas_diagnostico].astype(str).apply(lambda x: x.str.strip())

    # --- FUNCIÓN VECTORIZADA AUXILIAR ---
    def evaluar_prefijos(lista_prefijos):
        # Unimos los prefijos en un patrón Regex de inicio de cadena (ej: "^(X60|X61|X62)")
        # Usamos (?: ) para indicarle a Pandas que NO queremos capturar grupos de texto
        patron = "^(?:" + "|".join(lista_prefijos) + ")"
        # Evaluamos el patrón en todas las columnas simultáneamente (retorna matriz True/False)
        matriz_coincidencias = df_temp.apply(lambda col: col.str.contains(patron, regex=True, na=False))
        # Si al menos una columna de la fila es True, el resultado es 1, de lo contrario 0
        return matriz_coincidencias.any(axis=1).astype(int)

    # --- Bloque 1: Criterios de Inclusión (El Evento) ---
    prefijos_suicidio = [f'X{i}' for i in range(60, 85)] 
    df_EH['EH_intento_suicidio'] = evaluar_prefijos(prefijos_suicidio)
    df_EH['EH_ideacion_suicida'] = evaluar_prefijos(['R45.8'])
    df_EH['EH_antecedente_autolesion'] = evaluar_prefijos(['Z91.5'])

    # --- Bloque 2: Entorno Social y Económico ---
    df_EH['EH_vulnerabilidad_laboral'] = evaluar_prefijos(['Z56'])
    df_EH['EH_vulnerabilidad_economica'] = evaluar_prefijos(['Z59'])
    df_EH['EH_conflicto_soporte_familiar'] = evaluar_prefijos(['Z60', 'Z63'])
    df_EH['EH_historia_trauma_infantil'] = evaluar_prefijos(['Z61', 'Z62'])

    # --- Bloque 3: Salud Mental (Eje Psiquiátrico) ---
    df_EH['EH_trastorno_depresivo'] = evaluar_prefijos(['F32', 'F33'])
    df_EH['EH_trastorno_bipolar'] = evaluar_prefijos(['F31'])
    df_EH['EH_consumo_sustancias'] = evaluar_prefijos([f'F{i}' for i in range(10, 20)])
    df_EH['EH_trastorno_personalidad'] = evaluar_prefijos(['F60'])
    df_EH['EH_ansiedad_y_estres'] = evaluar_prefijos([f'F{i}' for i in range(40, 49)])

    # --- Bloque 4: Salud Física y Dolor Crónico ---
    prefijos_cancer = [f'C{str(i).zfill(2)}' for i in range(0, 98)] + [f'D{str(i).zfill(2)}' for i in range(0, 49)]
    df_EH['EH_enfermedad_oncologica'] = evaluar_prefijos(prefijos_cancer)

    # --- Bloque 5: Tiempos de Internación ---
    # Cambiado a coerce para evitar caídas si hay strings corruptos o vacíos en las fechas
    fecha_ingreso = pd.to_datetime(df_EH['Fecha ingreso'], format='%d/%m/%Y', errors='coerce')
    fecha_egreso = pd.to_datetime(df_EH['Fecha Egreso'], format='%d/%m/%Y', errors='coerce')
    
    df_EH['dias_internacion_'] = (fecha_egreso - fecha_ingreso).dt.days

    # Opcional: Reemplazar impresiones gigantescas por información de resumen en datasets reales
    # print(df_EH.to_string())
    return df_EH
