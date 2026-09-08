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
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--message', type=str, default='/home/carbajal/Documents/SaludMental/protocolo_mensaje_tipos.xml')
    parser.add_argument( '--classifiers', nargs='+', choices=['RandomForest', 'XGBoost', 'LogisticRegression', 'DecisionTree'],
                                                    default=['RandomForest']
)
    
    parser.add_argument('--log_comet', action='store_true')
    parser.add_argument('--combine', action='store_true', help='combine classifiers')
    #parser.add_argument('--split', type=str, default='estratificado')
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Nota: el --path es donde está la base de datos
    # el --working_dir puede estar en cualquier lado, pero con la siguiente estructura:
    #  ./working_dir/
    #      models/
    #      results/
    #      temp_files/

    args = parseCommandLineArguments()

    # ---------------- LOAD CONFIG ----------------
    config = load_config(args.config)
    filepath = config["data"].get('filepath')

    target = config["data"].get('target')  

    with open(args.message, "r", encoding="utf-8") as f:
        message = f.read()


    print(message)
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

    ################################################

    if target=='CAT_SUI_':
        print('Agregando los datos por persona...')

        df_IAE_agregada = agregar_base_intentos(df_IAE, dataset=2)




    # leo del archivo de configuración los atributos a usar 
    num_attribs = config["data"].get('num_features')
    cat_attribs = config["data"].get('cat_features')
    atributos = {'numericos':num_attribs, 'categoricos': cat_attribs}
   
    # obtengo los features que se usan para entrenar
    if target=='CAT_SUI_':
        X = getX(df_IAE_agregada, num_attribs, cat_attribs)
    else:
        X = getX(df_IAE, num_attribs, cat_attribs)

    #y = getY(data, metodo=args.y_method)

    print(X.info())
    print(X.shape)  

    

    root = ET.Element("Predictions")

    for clf_name in args.classifiers:
        model_name = f'{clf_name}_{target}'

        gs_clf = joblib.load(f'./modelos/{model_name}.joblib')
        estimator = gs_clf.best_estimator_

        predictions = estimator.predict_proba(X)

        classifier_node = ET.SubElement(
            root,
            "Classifier",
            name=model_name
        )

        ET.SubElement(
            classifier_node,
            "Probability"
        ).text = str(float(predictions[0, 1]))

    xml_string = ET.tostring(root, encoding="unicode")

    print(xml_string)

