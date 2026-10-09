import argparse 
import pandas as pd

from utils.utils import discretizar, conservar_filas_con_n_no_nulos, get_filas_con_n_nulos, eliminar_duplicados, convertir_enteros_a_fecha, eliminar_sin_cedula

from utils.IAE import preprocesar_IAE, agregar_tiempo_reincidencia
from utils.CNV import preprocesar_CNV
from utils.RUCAF import preprocesar_RUCAF, agregar_datos_RUCAF_en_IAE
from utils.SHARP import esta_persona_en_SHARPS
from utils.SIV import agregar_campos_SIV    
from utils.EH import agregar_datos_EH_cuando_intento, generar_indicadores_EH
from utils.agregado import agregar_base_intentos

from utils.load_data import load_config

import xml.etree.ElementTree as ET
import joblib

def getX(data, num_attribs, cat_attribs):
    # Este data frame (77 casos) incluye personas que tuvieron IAE y además murieron en 2023 (por suicidio u otras causas)

    # Evitar modificar los argumentos originales
    num_attribs = num_attribs.copy()
    cat_attribs = cat_attribs.copy() if cat_attribs else []
        
    if 'Sexo' in num_attribs:
        data['Sexo']=data['Sexo']=='Masculino'
    if 'PERSONA' in num_attribs:
        data['PERSONA']=data['PERSONA']=='Masculino'    
    if 'IAE_PREVIO_CORREGIDO_' in num_attribs:
        data['IAE_PREVIO_SI']=data['IAE_PREVIO_CORREGIDO_']=='SI'
        data['IAE_PREVIO_NO']=data['IAE_PREVIO_CORREGIDO_']=='NO'
        num_attribs.remove('IAE_PREVIO_CORREGIDO_')
        num_attribs.append('IAE_PREVIO_SI')
        num_attribs.append('IAE_PREVIO_NO')
    if 'RUCAF_cobertura' in num_attribs:
        data['RUCAF_cobertura_fonasa']=data['RUCAF_cobertura']=='Fonasa'
        data['RUCAF_cobertura_no_fonasa']=data['RUCAF_cobertura']=='No Fonasa'
        num_attribs.remove('RUCAF_cobertura')
        num_attribs.append('RUCAF_cobertura_fonasa')
        num_attribs.append('RUCAF_cobertura_no_fonasa')
    if 'CNV_otro_progenitor_' in num_attribs:  
        data['CNV_otro_progenitor_']=~ data['CNV_otro_progenitor_'].isnull()
    if 'ANTIPOLIOMELITICA' in num_attribs:  
        sin_vacuna = data['ANTIPOLIOMELITICA'].isnull()  
        data.loc[sin_vacuna,'ANTIPOLIOMELITICA']=0  
    if 'COVID 19' in num_attribs:  
        sin_vacuna = data['COVID 19'].isnull()  
        data.loc[sin_vacuna,'COVID 19']=0  

    X = data[num_attribs+cat_attribs] if cat_attribs else data[num_attribs]

    
    return X    





def parse_value(value, field_type):
    """
    Convierte un valor según el tipo especificado en el XML.
    """

    # Campo vacío
    if value is None or value.strip() == "":
        if field_type == "date":
            return pd.NaT
        elif field_type == "year-month":
            return pd.NaT
        else:
            return None

    value = value.strip()

    if field_type == "string":
        return value

    elif field_type == "integer":
        return int(value)

    elif field_type == "float":
        return float(value)

    elif field_type == "date":
        return pd.to_datetime(
            value,
            format="%Y-%m-%d",
            errors="coerce"
        )

    elif field_type == "year-month":
        return pd.to_datetime(
            value,
            format="%Y-%m",
            errors="coerce")
    else:
        raise ValueError(f"Tipo de campo desconocido: {field_type}")


def parse_message(message):
    """
    Parsea un mensaje XML recibido como string y convierte
    cada campo según el atributo 'type'.
    """

    root = ET.fromstring(message)

    result = {}

    for section in root.findall("section"):

        section_name = section.get("name")
        result[section_name] = []

        for record in section.findall("record"):

            data = {}

            for field in record.findall("field"):

                key = field.get("name")
                field_type = field.get("type")
                value = field.text

                data[key] = parse_value(value, field_type)

            result[section_name].append(data)

    return result



def parseCommandLineArguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, default='config/default.yaml')

    parser.add_argument(
        '--message',
        type=str,
        required=True,
        #default='/home/carbajal/Documents/SaludMental/protocolo_mensaje_tipos.xml',
        help='Archivo XML con los datos para realizar la inferencia'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Nombres de los modelos a utilizar, sin extensión'
    )

    #parser.add_argument( '--classifiers', nargs='+', choices=['RandomForest', 'XGBoost', 'LogisticRegression', 'DecisionTree'],
    #                                                default=['RandomForest']
    
    parser.add_argument('--log_comet', action='store_true')
    parser.add_argument('--combine', action='store_true', help='combine classifiers')
    #parser.add_argument('--split', type=str, default='estratificado')
 
    args = parser.parse_args()
    return args


def preprocesar_mensaje(message):

    print('Parseando mensaje...')
    data = parse_message(message)
    #print(data)

    df_IAE = pd.DataFrame(data['IAE'])
    df_RUCAF = pd.DataFrame(data['RUCAF'])
    df_CNV = pd.DataFrame(data['CNV'])
    df_SIV = pd.DataFrame(data['SIV'])
    df_EH = pd.DataFrame(data['EH'])
    df_SHARPS = pd.DataFrame(data['SHARPS'])

    print('Procesando IAE...')
    df_IAE = eliminar_sin_cedula(df_IAE)   
    df_IAE = eliminar_duplicados(df_IAE)   
        
   
    df_IAE = preprocesar_IAE(df_IAE)

    print('Procesando CNV...')
    df_IAE = preprocesar_CNV(df_CNV, df_IAE)

    ################ RUCAF #####################
    print('Procesando RUCAF...')

    df_RUCAF = preprocesar_RUCAF(df_RUCAF)

    df_RUCAF_grouped = df_RUCAF.groupby('cedula')
    df_IAE = df_IAE.apply( lambda row: agregar_datos_RUCAF_en_IAE(row, df_RUCAF_grouped), axis=1)
    
   ################# SHARPS  ######################
    print('Procesando SHARPS...')
    df_SHARPS = conservar_filas_con_n_no_nulos(df_SHARPS, 2)
    df_IAE = esta_persona_en_SHARPS(df_SHARPS, df_IAE)

    ################  SIV   #######################
    print('Procesando SIV...')
    df_SIV = conservar_filas_con_n_no_nulos(df_SIV, 2)
    df_SIV = eliminar_duplicados(df_SIV)
    df_IAE = agregar_campos_SIV(df_SIV, df_IAE)

    ################  EH  #########################
    print('Procesando EH...')
    df_EH = generar_indicadores_EH(df_EH)
    df_EH_agrupada = df_EH.groupby('cedula')
    df_IAE = df_IAE.apply( lambda row: agregar_datos_EH_cuando_intento(row, df_EH_agrupada), axis=1)

    print(df_IAE.keys())

    return df_IAE


from pathlib import Path

def realizar_inferencia(df_IAE, models):

    models_dir = Path("./modelos")

    root = ET.Element("Predictions")

    # Generar datos agregados por persona solamente si son necesarios
    df_IAE_agregada = None

    for model_name in models:

        print(f'Procesando modelo: {model_name}')

        # Cargar modelo calibrado
        model_path = models_dir / f"{model_name}_calib.joblib"
        calibrated_clf = joblib.load(model_path)

        # Recuperar configuración del modelo
        config = calibrated_clf.config

        target = config["data"]["target"]
        num_attribs = config["data"]["num_features"]
        cat_attribs = config["data"]["cat_features"]

        # Preprocesamiento específico según el target
        if target == 'CAT_SUI_':

            if df_IAE_agregada is None:
                print('Agregando datos por persona...')
                df_IAE_agregada = agregar_base_intentos(
                    df_IAE.copy(),
                    dataset=2
                )

            df_model = df_IAE_agregada

        else:
            df_model = df_IAE

        print("Target:", target)

        print("Atributos relacionados con IAE_PREVIO:")
        print([col for col in df_model.columns if "IAE_PREVIO" in col])

        print("Atributos faltantes:")
        features = num_attribs + (cat_attribs or [])
        print(set(features) - set(df_model.columns))
        # Construir atributos del modelo
        X = getX(df_model, num_attribs, cat_attribs)

        print(f'Dimensiones de X: {X.shape}')

        # Realizar predicciones
        probabilities = calibrated_clf.predict_proba(X)[:, 1]

        # Agregar resultados al XML
        classifier_node = ET.SubElement(
            root,
            "Classifier",
            name=model_name
        )

        for i, probability in enumerate(probabilities):

            prediction_node = ET.SubElement(
                classifier_node,
                "Prediction",
                index=str(i)
            )

            ET.SubElement(
                prediction_node,
                "Probability"
            ).text = str(float(probability))

        ET.SubElement(
            classifier_node,
            "Prevalence"
        ).text = str(float(calibrated_clf.prevalence))

    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":

    args = parseCommandLineArguments()

    # ---------------- LOAD CONFIG ----------------
    #config = load_config(args.config)
    #filepath = config["data"].get('filepath')

    #target = config["data"].get('target')  

    with open(args.message, "r", encoding="utf-8") as f:
        message = f.read()


    print(message)
    df_IAE = preprocesar_mensaje(message)

    ################################################

    # Inferencia con todos los modelos solicitados
    xml_predictions = realizar_inferencia(
        df_IAE,
        args.models
    )

    print(xml_predictions)
