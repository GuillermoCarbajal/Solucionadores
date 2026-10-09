#!/bin/bash

set -e

CONFIGS=(
    "entrega2.yaml"
    "entrega2_reintento.yaml"
    "entrega2_reintento_2d.yaml"
    "entrega2_reintento_10d.yaml"
    "entrega2_reintento_30d.yaml"
    "entrega2_reintento_60d.yaml"
)

for config in "${CONFIGS[@]}"; do

    echo "========================================"
    echo "Entrenando con configuración: $config"
    echo "========================================"

    python main.py \
        --config "config/$config" \
        --classifiers RandomForest XGBoost LogisticRegression DecisionTree \
        --gs_criteria roc_auc --log_comet 

done

echo "Todos los entrenamientos finalizaron."
