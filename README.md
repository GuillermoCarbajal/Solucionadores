Este repositorio contiene el código para entrenar modelos de clasificación y realizar inferencia a partir de datos recibidos mediante mensajes XML.

Se implementan cuatro clasificadores:

- Random Forest
- Regresión logística
- Árbol de decisión

El sistema permite entrenar modelos para distintas variables objetivo, realizar búsqueda de hiperparámetros mediante validación cruzada y calibrar las probabilidades de predicción.
# Instalación

```python
pip install -r requirements.txt
```

Se sugiere realizar la instalación en un entorno de Python creado con conda o venv para evitar conflictos con otras instalaciones. El motor fue desarrollado en Python 3.12. 


# Inferencia con modelos ya entrenados

El script `predict.py` permite generar predicciones utilizando los modelos previamente entrenados a partir de los datos recibidos en un mensaje XML.

Cada modelo calibrado contiene su propia configuración, por lo que no es necesario proporcionar un archivo YAML durante la inferencia.

El proceso incluye:

1. Lectura e interpretación del mensaje XML.
2. Preprocesamiento e integración de los datos.
3. Carga de los modelos solicitados.
4. Recuperación de la configuración asociada a cada modelo.
5. Construcción de los atributos correspondientes.
6. Generación de probabilidades calibradas.
7. Generación de un mensaje XML con las predicciones.

### 1.1. Parámetros de ejecución

| Argumento | Obligatorio | Descripción |
|---|---|---|
| `--message` | Sí | Ruta al archivo XML que contiene los datos sobre los que se realizará la inferencia. |
| `--models` | Sí | Nombres de los modelos que se utilizarán, sin la extensión `.joblib` ni el sufijo `_calib`. Se pueden especificar uno o varios. |

### 1.2. Ejemplos de ejecución

**Realizar inferencia utilizando un modelo:**

```bash
python predict.py \
    --message datos/mensaje.xml \
    --models RandomForest_CAT_SUI_
```

**Realizar inferencia utilizando varios modelos:**

```bash
python predict.py \
    --message datos/mensaje.xml \
    --models RandomForest_CAT_SUI_ XGBoost_CAT_SUI_ LogisticRegression_CAT_SUI_
```

Los modelos deben encontrarse en el directorio `./modelos/`.

### 1.3. Preprocesamiento de los datos

El mensaje XML contiene información proveniente de las siguientes fuentes:

- IAE
- RUCAF
- CNV
- SIV
- EH
- SHARPS

Los datos se procesan e integran utilizando las funciones de preprocesamiento correspondientes a cada fuente.

Posteriormente, para cada modelo se recuperan de su configuración los atributos numéricos y categóricos necesarios para construir la matriz de entrada `X`.

**Preprocesamiento específico según el target:**

- **`CAT_SUI_`:** se utiliza `agregar_base_intentos()` para agregar los registros por persona antes de construir los atributos.
- **Otros targets:** se utilizan directamente los registros de la base IAE procesada, sin realizar la agregación anterior.

La función `getX()` realiza las transformaciones de atributos necesarias, de acuerdo con la configuración de cada modelo.

### 1.4. Carga de modelos

Para cada nombre recibido mediante `--models`, el programa carga:

```text
./modelos/{model_name}_calib.joblib
```

A partir del modelo se recuperan:

- El estimador entrenado.
- El calibrador de probabilidades.
- La prevalencia de la clase positiva en los datos de entrenamiento.
- La configuración utilizada durante el entrenamiento.

Esto permite utilizar modelos con diferentes variables objetivo y conjuntos de atributos en una misma ejecución.

### 1.5. Formato de salida

El programa genera un mensaje XML que contiene las predicciones de todos los modelos solicitados.

Para cada modelo se incluyen:

- **Classifier:** nombre del modelo.
- **Prediction:** predicción correspondiente a una observación.
- **Probability:** probabilidad calibrada de pertenecer a la clase positiva.
- **Prevalence:** prevalencia de la clase positiva en los datos de entrenamiento.

Ejemplo de salida para dos modelos y tres observaciones:

```xml
<Predictions>
    <Classifier name="RandomForest_CAT_SUI_">
        <Prediction index="0">
            <Probability>0.15</Probability>
        </Prediction>
        <Prediction index="1">
            <Probability>0.32</Probability>
        </Prediction>
        <Prediction index="2">
            <Probability>0.08</Probability>
        </Prediction>
        <Prevalence>0.05</Prevalence>
    </Classifier>
    <Classifier name="XGBoost_CAT_SUI_">
        <Prediction index="0">
            <Probability>0.12</Probability>
        </Prediction>
        <Prediction index="1">
            <Probability>0.28</Probability>
        </Prediction>
        <Prediction index="2">
            <Probability>0.09</Probability>
        </Prediction>
        <Prevalence>0.05</Prevalence>
    </Classifier>
</Predictions>
```

Los valores son ilustrativos.

Se genera una predicción por cada fila de la matriz de atributos correspondiente al modelo. Para `CAT_SUI_`, las filas corresponden a los registros agregados por persona; para los demás targets, corresponden a los registros procesados de IAE.

Actualmente, las predicciones se identifican mediante un índice que indica su posición en la matriz de entrada.

El XML generado se imprime en la salida estándar.


## Entrenamiento de modelos

El script `main.py` permite entrenar uno o varios clasificadores utilizando la configuración especificada en un archivo YAML.

El entrenamiento incluye:

1. Carga y filtrado de los datos.
2. Selección y transformación de atributos.
3. Búsqueda de hiperparámetros mediante Grid Search con validación cruzada.
4. Entrenamiento del modelo seleccionado.
5. Evaluación mediante predicciones de validación cruzada.
6. Calibración de las probabilidades de predicción.
7. Almacenamiento de los modelos y resultados.
   
### 2.1. Parámetros de ejecución

| Argumento | Valor por defecto | Descripción |
|---|---|---|
| `--config` | `config/default.yaml` | Ruta al archivo YAML que contiene la configuración del experimento. |
| `--classifiers` | `RandomForest` | Clasificadores a entrenar. Se pueden especificar uno o varios entre `RandomForest`, `XGBoost`, `LogisticRegression` y `DecisionTree`. |
| `--gs_criteria` | `roc_auc` | Métrica utilizada para seleccionar los mejores hiperparámetros durante Grid Search. |
| `--disable_class_weights`, `-dcw` | `False` | Desactiva la ponderación de clases utilizada para compensar el desbalance entre clases. |
| `--tsne` | `False` | Genera una visualización de los datos mediante t-SNE. |
| `--log_comet` | `False` | Habilita el registro de parámetros, métricas y gráficos en Comet. |
| `--combine` | `False` | Evalúa combinaciones de los clasificadores entrenados mediante `hard_voting`, `soft_voting` y `weighted_soft_voting`. |

Los argumentos booleanos se activan incluyéndolos en el comando, sin necesidad de indicar un valor.

### 2.2. Ejemplos de ejecución

**Entrenar un Random Forest con la configuración por defecto:**

```bash
python main.py
```

**Entrenar tres clasificadores utilizando una configuración específica:**

```bash
python main.py \
    --config config/entrega2.yaml \
    --classifiers RandomForest XGBoost LogisticRegression
```

**Entrenar los cuatro clasificadores y registrar los resultados en Comet:**

```bash
python main.py \
    --config config/entrega2.yaml \
    --classifiers RandomForest XGBoost LogisticRegression DecisionTree \
    --gs_criteria roc_auc \
    --log_comet
```

### 2.3. Entrenamiento de múltiples configuraciones

El script `train_all.sh` permite ejecutar secuencialmente los entrenamientos para las siguientes configuraciones:

- `entrega2.yaml`
- `entrega2_reintento.yaml`
- `entrega2_reintento_2d.yaml`
- `entrega2_reintento_10d.yaml`
- `entrega2_reintento_30d.yaml`
- `entrega2_reintento_60d.yaml`

Para ejecutarlo:

```bash
chmod +x train_all.sh
./train_all.sh
```

El script entrena los cuatro clasificadores para cada configuración, ejecutando un total de 24 entrenamientos.

Si alguno de los entrenamientos falla, la ejecución se detiene.

### 2.4. Archivos generados

Para cada clasificador y variable objetivo se generan los siguientes archivos:

| Archivo | Descripción |
|---|---|
| `{classifier}_{target}.joblib` | Modelo entrenado, incluyendo el preprocesamiento y los resultados de Grid Search. |
| `{classifier}_{target}_calib.joblib` | Modelo calibrado, que incluye el estimador entrenado, el calibrador, la prevalencia y la configuración utilizada durante el entrenamiento. |
| `{classifier}_{target}_train_predictions.csv` | Predicciones del modelo calibrado sobre los datos de entrenamiento. |

El archivo de configuración YAML también puede conservarse como copia independiente junto con los modelos.

**Importante:** los nombres de los archivos se construyen utilizando el clasificador y la variable objetivo. Si se entrenan dos modelos con el mismo clasificador y target, sus archivos pueden sobrescribirse.


# Preprocesamiento de los datos


Antes de entrenar los modelos es necesario procesar e integrar las diferentes bases de datos disponibles. Este procedimiento se realiza mediante el script `main_preprocesar.py`.

El preprocesamiento genera dos bases de datos: una que conserva los registros individuales de intentos de autoeliminación (IAE) y otra que agrega la información por persona. Estas bases se utilizan posteriormente para entrenar los modelos de clasificación según la variable objetivo.

### 1.1. Parámetros de ejecución

| Argumento | Valor por defecto | Descripción |
|---|---|---|
| `--path` | `/home/carbajal/Documents/SaludMental/2da entrega 20260210/Planilla completa.xlsx` | Ruta al archivo Excel que contiene las bases de datos originales. |

Actualmente, el código está configurado para procesar la segunda entrega de datos (`dataset = 2`). Esta entrega incorpora la base de egresos hospitalarios (EH), que no está disponible en la primera.

### 1.2. Ejemplo de ejecución

Para procesar las bases de datos:

```bash
python main_preprocesar.py \
    --path "datos/Planilla completa.xlsx"
```

### 1.3. Procesamiento de las bases de datos

El script carga las diferentes bases de datos y realiza tareas de limpieza, transformación de atributos e integración de información.

La base IAE se utiliza como base principal, a la que se incorporan atributos provenientes de las demás fuentes.

| Base | Procesamiento realizado |
|---|---|
| **IAE** | Eliminación de registros sin cédula y registros duplicados. Limpieza y transformación de atributos mediante `preprocesar_IAE()`. |
| **CDE** | Eliminación de registros con menos de cuatro campos no nulos. Generación de las variables objetivo `CAT_SUI` y `CAT_MCEXSUI`, análisis de edades al fallecimiento y discretización de edades en intervalos de cinco años. Incorporación de información de defunciones a IAE y cálculo de tiempos de reincidencia. |
| **CNV** | Procesamiento e incorporación de información de nacimientos mediante `preprocesar_CNV()`. |
| **RUCAF** | Limpieza y transformación de los datos mediante `preprocesar_RUCAF()`. Agrupación por cédula e incorporación de información de cobertura a los registros de IAE. |
| **SHARPS** | Eliminación de registros con menos de dos campos no nulos e incorporación de indicadores de presencia en SHARPS. |
| **SIV** | Eliminación de registros con menos de dos campos no nulos y de registros duplicados. Incorporación de información de SIV a IAE. |
| **EH** | Generación de indicadores a partir de los egresos hospitalarios e incorporación de información correspondiente a cada intento. Disponible para la segunda entrega. |

Durante el procesamiento también se realizan controles exploratorios, incluyendo análisis de valores faltantes, valores únicos, frecuencias y rangos de determinados atributos, así como verificaciones de correspondencia entre las diferentes bases.

### 1.4. Generación de las bases procesadas

Una vez integrados los datos, el script genera dos archivos CSV.

**Base sin agregar por persona**

```text
IAE_sin_agregar_entrega2_YYYY-MM-DD_HH-MM-SS.csv
```

Contiene los registros individuales de IAE, enriquecidos con la información proveniente de las demás bases.

Esta base permite trabajar con variables objetivo definidas a nivel de intento.

**Base agregada por persona**

```text
IAE_agregada_entrega2_YYYY-MM-DD_HH-MM-SS.csv
```

Se obtiene mediante la función `agregar_base_intentos()`, que agrega la información de los distintos intentos correspondientes a una misma persona.

Esta base se utiliza para entrenar modelos cuya variable objetivo está definida a nivel de persona, como `CAT_SUI_`.

Los nombres de ambos archivos incluyen una marca de tiempo para identificar cuándo fueron generados y evitar sobrescribir resultados de ejecuciones anteriores.

### 1.5. Relación con el entrenamiento y la inferencia

Los archivos CSV generados durante el preprocesamiento se utilizan como entrada del script de entrenamiento `main.py`. La ruta a la base correspondiente se especifica mediante el atributo `data.filepath` del archivo YAML de configuración.

Durante la inferencia, `predict.py` recibe los datos en formato XML y reproduce el preprocesamiento necesario para construir los atributos de entrada de cada modelo.

En particular:

- Para modelos con target `CAT_SUI_`, se realiza la agregación de registros por persona mediante `agregar_base_intentos()`.
- Para los demás targets, se utiliza la información procesada a nivel de intento.

En ambos casos, la función `getX()` construye la matriz de atributos utilizando la configuración almacenada en cada modelo calibrado.

Esto permite mantener la correspondencia entre los atributos utilizados durante el entrenamiento y los generados durante la inferencia.


