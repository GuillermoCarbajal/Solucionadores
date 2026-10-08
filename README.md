## Instalación

```python
pip install -r requirements.txt
```

Se sugiere realizar la instalación en un entorno de Python creado con conda o venv para evitar conflictos con otras instalaciones. El motor fue desarrollado en Python 3.12. 

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

### Ejecución de modelos ya entrenados

```python
python predict.py --config config/entrega2_reintento.yaml --message mensaje_en_xml --classifiers LogisticRegression DecisionTree RandomForest
```
