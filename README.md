## Instalación

```python
pip install requirements.txt
```

### Entrenamiento

```python
python main.py --config config/entrega2_reintento.yaml --classifiers LogisticRegression DecisionTree RandomForest
```

### Preprocesamiento de los datos

```python
python main_preprocesar.py
```

### Ejecución de modelos ya entrenados

```python
python predict.py --config config/entrega2_reintento.yaml --message mensaje_en_xml --classifiers LogisticRegression DecisionTree RandomForest
```
