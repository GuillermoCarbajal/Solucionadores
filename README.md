## Instalación

```python
pip install -r requirements.txt
```

Se sugiere realizar la instalación en un entorno de Python creado con conda o venv para evitar conflictos con otras instalaciones. El motor fue desarrollado en Python 3.12. 

### Entrenamiento

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
