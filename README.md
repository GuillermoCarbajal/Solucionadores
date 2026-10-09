## Instalación

```python
pip install -r requirements.txt
```

Se sugiere realizar la instalación en un entorno de Python creado con conda o venv para evitar conflictos con otras instalaciones. El motor fue desarrollado en Python 3.12. 


## Inferencia

El script de inferencia permite cargar modelos previamente entrenados y generar predicciones a partir de los datos recibidos en un mensaje XML.

El proceso incluye el preprocesamiento de los datos, la construcción de los atributos utilizados por los modelos y la generación de probabilidades calibradas.

### Argumentos disponibles

| Argumento | Valor por defecto | Descripción |
|---|---|---|
| `--config` | `config/default.yaml` | Ruta al archivo YAML que contiene la configuración del modelo, incluyendo la variable objetivo y los atributos utilizados. |
| `--message` | `/home/carbajal/Documents/SaludMental/protocolo_mensaje_tipos.xml` | Ruta al archivo XML que contiene los datos sobre los que se realizará la inferencia. |
| `--classifiers` | `RandomForest` | Clasificadores a utilizar. Se pueden especificar uno o varios entre `RandomForest`, `XGBoost`, `LogisticRegression` y `DecisionTree`. |
| `--log_comet` | `False` | Argumento definido, pero actualmente no utilizado en el proceso de inferencia. |
| `--combine` | `False` | Argumento definido, pero actualmente no utilizado en el proceso de inferencia. |

### Ejemplos de ejecución

**Realizar inferencia utilizando una configuración y un mensaje XML específicos:**

```bash
python predict.py \
    --config config/default.yaml \
    --message datos/mensaje.xml
```

**Realizar inferencia utilizando varios clasificadores:**

```bash
python predict.py \
    --config config/default.yaml \
    --message datos/mensaje.xml \
    --classifiers RandomForest XGBoost LogisticRegression
```

### Procesamiento de los datos

El script recibe un mensaje XML que contiene información de las siguientes fuentes:

- **IAE:** registros de intentos de autoeliminación.
- **RUCAF**
- **CNV**
- **SIV**
- **EH**
- **SHARPS**

Los datos son procesados utilizando las mismas funciones de preprocesamiento empleadas durante la preparación de los datos de entrenamiento.

Posteriormente, se construye la matriz de atributos `X` utilizando las variables numéricas y categóricas especificadas en el archivo de configuración.

Cuando la variable objetivo es `CAT_SUI_`, se realiza previamente una agregación de los registros por persona.

### Carga de modelos

Los modelos deben encontrarse en el directorio `./modelos/`.

Para cada clasificador se cargan dos archivos:

| Archivo | Descripción |
|---|---|
| `{classifier}_{target}.joblib` | Modelo entrenado, incluyendo el preprocesamiento y la selección de hiperparámetros. |
| `{classifier}_{target}_calib.joblib` | Modelo calibrado utilizado para obtener las probabilidades finales. |

El valor de `target` se obtiene del archivo de configuración y debe coincidir con el utilizado durante el entrenamiento.

### Formato de salida

El script genera un mensaje XML con las predicciones de los clasificadores seleccionados.

Para cada clasificador se incluyen:

- **Classifier:** nombre del clasificador y variable objetivo.
- **Probability:** probabilidad calibrada de pertenecer a la clase positiva.
- **Prevalence:** prevalencia de la clase positiva en los datos utilizados para entrenar el modelo.

Ejemplo de salida:

```xml
<Predictions>
    <Classifier name="RandomForest_CAT_SUI_">
        <Probability>0.15</Probability>
        <Prevalence>0.08</Prevalence>
    </Classifier>
    <Classifier name="XGBoost_CAT_SUI_">
        <Probability>0.12</Probability>
        <Prevalence>0.08</Prevalence>
    </Classifier>
</Predictions>
```

Los valores anteriores son ilustrativos.

Actualmente, el script devuelve la probabilidad correspondiente a la primera fila de la matriz de atributos (`X`). Por lo tanto, la salida está diseñada para obtener una predicción por clasificador.

El XML generado se imprime en la salida estándar y no se guarda automáticamente en un archivo.


## Entrenamiento


### Parámetros de ejecución

El script permite configurar el entrenamiento y la evaluación de los clasificadores mediante argumentos de línea de comandos.

#### Argumentos disponibles

| Argumento | Valor por defecto | Descripción |
|---|---|---|
| `--config` | `config/default.yaml` | Ruta al archivo YAML que contiene la configuración del experimento. |
| `--classifiers` | `RandomForest` | Clasificadores a entrenar. Se pueden especificar uno o varios entre `RandomForest`, `XGBoost`, `LogisticRegression` y `DecisionTree`. |
| `--gs_criteria` | `roc_auc` | Métrica utilizada para seleccionar los mejores hiperparámetros durante la búsqueda mediante Grid Search. |
| `--disable_class_weights`, `-dcw` | `False` | Desactiva la ponderación de clases utilizada para compensar el desbalance entre clases. |
| `--tsne` | `False` | Genera una visualización de los datos mediante t-SNE. |
| `--log_comet` | `False` | Habilita el registro de parámetros, métricas y gráficos del experimento en Comet. |
| `--combine` | `False` | Evalúa combinaciones de los clasificadores entrenados mediante votación mayoritaria (`hard_voting`), promedio de probabilidades (`soft_voting`) y promedio ponderado (`weighted_soft_voting`). |

Los argumentos booleanos (`--disable_class_weights`, `--tsne`, `--log_comet` y `--combine`) se activan simplemente incluyéndolos en el comando, sin necesidad de indicar un valor.

#### Ejemplos de ejecución

**Entrenar un clasificador Random Forest con la configuración por defecto:**

```bash
python main.py
```

**Entrenar tres clasificadores utilizando una configuración específica:**

```bash
python main.py \
    --config config/default.yaml \
    --classifiers RandomForest XGBoost LogisticRegression
```

**Entrenar tres clasificadores, optimizando ROC AUC y registrando los resultados en Comet:**

```bash
python main.py \
    --classifiers RandomForest XGBoost LogisticRegression \
    --gs_criteria roc_auc \
    --log_comet
```

**Entrenar varios clasificadores sin ponderación de clases y evaluar sus combinaciones:**

```bash
python main.py \
    --classifiers RandomForest XGBoost LogisticRegression \
    --disable_class_weights \
    --combine
```

### Modelos generados

Para cada clasificador entrenado se generan los siguientes archivos:

| Archivo | Descripción |
|---|---|
| `{classifier}_{target}.joblib` | Modelo entrenado, incluyendo el preprocesamiento y la búsqueda de hiperparámetros. |
| `{classifier}_{target}_calib.joblib` | Modelo con probabilidades calibradas. |
| `{classifier}_{target}_train_predictions.csv` | Predicciones del modelo calibrado sobre los datos de entrenamiento. |
| `{classifier}_{target}.yaml` | Copia del archivo de configuración utilizado para entrenar el modelo (si se implementó el guardado de configuración). |

Donde `classifier` identifica el tipo de clasificador y `target` corresponde a la variable objetivo definida en el archivo de configuración.


```python
python main.py --config config/entrega2_reintento.yaml --classifiers LogisticRegression DecisionTree RandomForest
```


### Preprocesamiento de los datos

```python
python main_preprocesar.py --path path/to/2da entrega 20260210/Planilla completa.xlsx
```
El entrenamiento se realizó con los datos de la segunda entrega.


