from utils.comet_utils import create_experiment, save_gs_results, save_cv_results
from utils.load_data import load_config
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import HistGradientBoostingClassifier 

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


from sklearn.model_selection import KFold
from sklearn.base import clone

from utils.visualization import mostrar_tsne
from utils.evaluacion import oof_prediction

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

import argparse

os.environ["COMET_AUTO_LOGGING"] = "0"

operacion = {
    "igual_a": lambda s, v: s == v,
    "distinto_a": lambda s, v: s != v,
    "mayor_que": lambda s, v: s > v,
    "mayor_o_igual_que": lambda s, v: s >= v,
    "menor_que": lambda s, v: s < v,
    "menor_o_igual_que": lambda s, v: s <= v,
    "no_nulo": lambda s, v: s.notna(),
    "es_nulo": lambda s, v: s.isna(),
}

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

def getY(data, metodo='defuncion'):
    # Este data frame (77 casos) incluye personas que tuvieron IAE y además murieron en 2023 (por suicidio u otras causas)
    if metodo=='defuncion':
        y = data["DEFUNCION_"]
    elif metodo=='30dias':
        y = data["MIN_DIAS_IAE_MUERTE_"]<30
    elif metodo=='60dias':
        y = data["MIN_DIAS_IAE_MUERTE_"]<60
    elif metodo=='90dias':
        y = data["MIN_DIAS_IAE_MUERTE_"]<90
    elif metodo=='CAT_SUI_':
        y = data["CAT_SUI_"]==1
    elif metodo=='CAT_MCEXSUI_':
        y = data["CAT_MCEXSUI_"]==1
    elif metodo=='MOTIVO_EXT_SUI_':
        y=data["MOTIVO_EXT_SUI_"]==1
    return y    



def generate_preprocessing_pipeline(num_attribs, cat_attribs, classifier=None):


    if classifier=='LogisticRegression':
        num_pipeline = Pipeline([
        ("imputer", SimpleImputer( strategy='mean')),
        ("standardize", StandardScaler()),
        ])

    else:
        num_pipeline = Pipeline([
        #("standardize", StandardScaler()),
        ("pass", 'passthrough'),
        ])

    cat_pipeline = make_pipeline(
        #SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    if cat_attribs:
        preprocessing = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs)
            ], remainder='passthrough')
    else:
        preprocessing = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ], remainder='passthrough')
        
    return preprocessing



def create_folds(metodo='default'):

    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import StratifiedGroupKFold


    if metodo=='default':
        cv = KFold(
            n_splits=5,
            shuffle=False
        )

    elif metodo=='estratificado':

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=False
        )
    elif metodo=='grupo_estratificado':

        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=False,
        )

    return cv

def build_model_with_cv(preprocessing, classifier='LogisticRegression', class_weights=None,cv=5, 
                        criteria='roc_auc', random_state=33, n_jobs=-1):
    """
    Construye un pipeline con preprocesamiento y optimización de hiperparámetros
    mediante validación cruzada, para distintos clasificadores.
    """

    # Modelos base
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000,class_weight=class_weights, random_state=random_state),
        'RandomForest': RandomForestClassifier(class_weight=class_weights, random_state=random_state),
        'XGBoost': XGBClassifier(tree_method="hist", objective='binary:logistic', eval_metric='aucpr', 
                                 enable_categorical=True, scale_pos_weight=class_weights,verbosity=2, random_state=random_state),
        'SVM': SVC(probability=True,class_weight=class_weights),
        'DecisionTree': DecisionTreeClassifier(class_weight=class_weights, random_state=random_state),
        'HistGradientBoosting': HistGradientBoostingClassifier(class_weight=class_weights, random_state=random_state),
    }

    # Espacios de búsqueda razonables por modelo
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__l1_ratio': [0],
            'classifier__solver': ['lbfgs']
        },
        'RandomForest': {
            'classifier__n_estimators': [25, 50, 100, 200],
            'classifier__max_depth': [3, 5, 10, 20, 50, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200, 500, 1000],
            'classifier__max_depth': [2, 3, 5, 8],
            'classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        },
        'DecisionTree': {
            'classifier__max_depth': [2, 3, 4, 5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'HistGradientBoosting': {
            'classifier__max_iter': [100, 200, 500, 1000],
            'classifier__max_depth': [3, 5, 8],
            'classifier__learning_rate': [0.001, 0.01, 0.1, 0.3]
        },

    }

    # Verificación
    if classifier not in classifiers:
        raise ValueError(f"Clasificador '{classifier}' no soportado. Opciones: {list(classifiers.keys())}")

    # Construcción del pipeline
    base_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifiers[classifier])
    ])
    
    print(base_pipeline.named_steps['classifier'])
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2),
        "pr_auc": "average_precision",  # proxy de PR-AUC
        "roc_auc": "roc_auc",  # proxy de PR-AUC
        "bal_acc": "balanced_accuracy"
    }

    # Configurar búsqueda con CV
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grids[classifier],
        cv=cv,
        n_jobs=n_jobs,
        refit=criteria,
        scoring=scoring,
        return_train_score=True, verbose=1
    )

    return grid_search



def show_grid_search_results(gs_pipeline, criteria):

    #for clase, scores in classifier.scores_.items():
    #    print(f"Clase {clase}: shape {scores.shape}")  # (n_folds, n_Cs)

    #scores = list(classifier.scores_.values())[0]  # si es binario, tomamos la clase positiva
    #mean_scores = np.mean(scores, axis=0)
    #std_scores = np.std(scores, axis=0)

    mean_scores = gs_pipeline.cv_results_[f'mean_test_{criteria}']
    std_scores = gs_pipeline.cv_results_[f'std_test_{criteria}']

    x = np.arange(len(mean_scores))

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, mean_scores, marker='o', label='Mean CV Score')
    ax.fill_between(
        np.arange(len(mean_scores)),
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2,
        label='±1 Std. Dev.'
    )
    #plt.xlabel("C (Inverse of Regularization Strength)")
    ax.set_ylabel(f"Mean Cross-Validation {criteria}")
    ax.set_title("Cross-Validation Performance")
    ax.legend()
    return fig
    #plt.show()



def show_best_cv_results(gs_pipeline, criteria):

    best_idx = gs_pipeline.best_index_
    cv_results = gs_pipeline.cv_results_
    n_splits = gs_pipeline.n_splits_

    train_scores = [
        cv_results[f'split{i}_train_{criteria}'][best_idx]
        for i in range(n_splits)
    ]

    test_scores = [
        cv_results[f'split{i}_test_{criteria}'][best_idx]
        for i in range(n_splits)
    ]

    folds = np.arange(1, n_splits + 1)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(folds, train_scores, 'o-', label='Train')
    ax.plot(folds, test_scores, 'o-', label='Validation')

    ax.axhline(np.mean(train_scores), ls='--', color='b', label='mean train')
    ax.axhline(np.mean(test_scores), ls='--', color='r', label='mean val')

    ax.set_xlabel('Fold')
    ax.set_ylabel(criteria)
    ax.set_title('Best hyperparameter setting')
    ax.legend()

    return fig

def show_cv_confusion_matrix(cv_scores, y, threshold=0.5, titulo='Matriz de confusion', filename=None):

    fig, ax = plt.subplots()
    # Calcular la matriz de confusión
    cm = confusion_matrix(y, cv_scores>threshold)

    # Mostrarla
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{titulo}, th={threshold:.02f}')
    #plt.savefig('CV_conf_matrix.png' if filename is None else filename)
    #plt.show()
    return fig

def show_cv_precision_recall(cv_scores, y, filename=None, titulo='Curva Prec. Recall (val cruzada)', save_path=None):
    
    precision, recall, _ = precision_recall_curve(y, cv_scores)
    ap = average_precision_score(y, cv_scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.6)

    if save_path:
        fig.savefig(os.path.join(save_path,'CV_PR.png') if filename is None else os.path.join(save_path,filename))
    
    #plt.show()
    return fig


import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def log_calibration_analysis(
    y_score,
    y_true,
    name="model",
    n_bins=10,
    experiment=None
):
    """
    Loguea en Comet:
      - Reliability diagram
      - Histograma de scores por clase
      - Brier score
      - Log loss

    Parameters
    ----------
    y_true : array-like
        Etiquetas verdaderas (0/1)
    y_score : array-like
        Probabilidades de la clase positiva
    name : str
        Prefijo para nombres de figuras y métricas
    n_bins : int
        Número de bins para la curva de calibración
    experiment : comet_ml.Experiment
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # métricas
    brier = brier_score_loss(y_true, y_score)
    ll = log_loss(y_true, y_score)

    if experiment:
        experiment.log_metric(f"{name}_brier_score", brier)
        experiment.log_metric(f"{name}_log_loss", ll)

    # curva de calibración
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy="quantile"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # -------------------------------------------------
    # Reliability diagram
    # -------------------------------------------------

    ax = axes[0]

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Perfect calibration"
    )

    ax.plot(
        mean_pred,
        frac_pos,
        "o-",
        label=name
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(
        f"Calibration curve\n"
        f"Brier={brier:.4f}  LogLoss={ll:.4f}"
    )

    ax.legend()
    ax.grid(True)

    # -------------------------------------------------
    # Histograma de scores
    # -------------------------------------------------

    ax = axes[1]

    ax.hist(
        y_score[y_true == 0],
        bins=20,
        alpha=0.5,
        label="Negative"
    )

    ax.hist(
        y_score[y_true == 1],
        bins=20,
        alpha=0.5,
        label="Positive"
    )

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution")
    ax.legend()

    plt.tight_layout()

    fig.canvas.draw()

    if experiment:
        experiment.log_figure(
            figure_name=f"{name}_calibration_analysis",
            figure=fig
        )

    plt.close(fig)

def show_cv_roc(cv_scores, y, filename=None, titulo='Curva ROC', save_path=None):

     #y_score = model1_pipeline.predict_proba(data1_X)[:, 1]  # probabilidad de la clase positiva
    fpr, tpr, _ = roc_curve(y, cv_scores)
    auc = roc_auc_score(y, cv_scores)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(titulo)
    ax.legend()

    if save_path:
        fig.savefig(os.path.join(save_path,'CV_PR.png') if filename is None else os.path.join(save_path,filename))

    #plt.savefig('Cross Validated ROC curve'  if filename is None else filename)
    #plt.show()
    return fig


def conflicting_patterns_counts(X, y, round_decimals=None):
    # convertir a DataFrame si hace falta
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    df = X.copy().reset_index(drop=True)
    df["_y_"] = pd.Series(y).values

    # optional: redondear para evitar pequeñas diferencias float
    if round_decimals is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.drop("_y_", errors="ignore")
        df[num_cols] = df[num_cols].round(round_decimals)

    # 1) Agrupamos por todas las columnas de X y por la etiqueta, y contamos
    grp = df.groupby(list(X.columns) + ["_y_"]).size().rename("n").reset_index()

    # 2) Hacemos pivot para tener por fila-patrón las columnas por clase
    pivot = grp.pivot_table(index=list(X.columns), columns="_y_", values="n", fill_value=0)

    # 3) Validación: la suma de todos los conteos en pivot debe ser igual a len(df)
    total_rows = len(df)
    total_from_pivot = int(pivot.values.sum())
    print(f"Total filas en df: {total_rows}; Suma de counts en pivot: {total_from_pivot}")
    if total_rows != total_from_pivot:
        print("⚠️ Inconsistencia detectada: la suma de counts en pivot NO coincide con el número total de filas.")
        print("A continuación se muestran las primeras filas de 'grp' para inspección:")
        print(grp.head(20))
        # devolver resultados parciales para inspección
        return {"pivot": pivot, "grp": grp, "df": df, "consistent": False}

    # 4) Filtrar solo patrones que tienen más de una clase presente (conflictos)
    conflicts = pivot[pivot.gt(0).sum(axis=1) > 1]

    # 5) Añadir total por patrón y ordenar por total descendente
    if not conflicts.empty:
        conflicts["total"] = conflicts.sum(axis=1)
        conflicts = conflicts.sort_values("total", ascending=False)

    return {"pivot": pivot, "conflicts": conflicts, "consistent": True}




def mostrar_histogramas(X, y, nombres=None):
    clases = y.unique()
    num_cols = X.shape[1]

    if nombres is None:
        nombres = X.columns

    nrows = (num_cols - 1) // 4 + 1
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, (ax, col) in enumerate(zip(axes, range(num_cols))):
        serie = X.iloc[:, col]
        nombre = nombres[i]

        es_numerica = pd.api.types.is_numeric_dtype(serie)

        # -------- NUMÉRICA --------
        if es_numerica and serie.nunique() > 10:
            # bins globales
            bins = np.histogram_bin_edges(serie.dropna(), bins=10)

            for c in clases:
                datos = X[y == c].iloc[:, col]
                ax.hist(datos, bins=bins, density=True,
                        alpha=0.5, label=f"Clase {c}")

        # -------- CATEGÓRICA / BINARIA --------
        else:
            categorias = serie.dropna().unique()
            categorias = sorted(categorias)

            x = np.arange(len(categorias))
            width = 0.8 / len(clases)  # barras lado a lado

            for j, c in enumerate(clases):
                datos = X[y == c].iloc[:, col]
                counts = datos.value_counts(normalize=True) #

                valores = [counts.get(cat, 0) for cat in categorias]

                ax.bar(x + j * width, valores,
                       width=width, alpha=0.7,
                       label=f"Clase {c}")

            ax.set_xticks(x + width * (len(clases) - 1) / 2)
            ax.set_xticklabels(categorias)

        # -------- FORMATO --------
        ax.set_title(nombre)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelrotation=45)

        for label in ax.get_xticklabels():
            label.set_ha('right')

        ax.legend()

    # borrar ejes vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return plt


def reportar_validacion_cruzada(cv_scores, y, threshold=0.5, cmt_exp=None, 
                                indices_subgrupo=[], nombre_subgrupo=''):
    
    y_eval = y[indices_subgrupo] if len(indices_subgrupo)>0 else y
    cv_scores_eval =cv_scores[indices_subgrupo] if len(indices_subgrupo)>0 else cv_scores

    precision = precision_score(y_eval, cv_scores_eval > threshold)
    accuracy = accuracy_score(y_eval, cv_scores_eval > threshold)
    recall = recall_score(y_eval, cv_scores_eval > threshold)
        
    ap = average_precision_score(y_eval, cv_scores_eval)
    roc_auc = roc_auc_score(y_eval, cv_scores_eval)
    titulo_mc = 'Matriz de Confusión (val. cruzada)' if nombre_subgrupo=='' else f'Matriz de Confusión (val. curzada) para {nombre_subgrupo}'
    fig_cm = show_cv_confusion_matrix(cv_scores_eval, y_eval, threshold, titulo=titulo_mc)

    titulo_pr = 'Curva Prec. Recall (val. cruzada)' if nombre_subgrupo=='' else f'Curva Prec. Recall (val. curzada) para {nombre_subgrupo}'
    fig_pr = show_cv_precision_recall(cv_scores_eval, y_eval, titulo=titulo_pr)
    
    titulo_roc = 'Curva ROC (val. cruzada)' if nombre_subgrupo=='' else f'Curva ROC (val. curzada) para {nombre_subgrupo}'
    fig_roc = show_cv_roc(cv_scores_eval,y_eval, titulo=titulo_roc)
    
 
    titulo_curva_calib = 'Curva de probabilidad (val. cruzada)' if nombre_subgrupo=='' else f'Curva de probabilidad (val. curzada) para {nombre_subgrupo}'
    log_calibration_analysis(y_score=cv_scores_eval, y_true=y_eval,  name=titulo_curva_calib, experiment=cmt_exp)

    calibr_transform = LogisticRegression()
    calibr_transform.fit( cv_scores.reshape(-1, 1), y)
    scores_eval_cal = calibr_transform.predict_proba(  cv_scores_eval.reshape(-1, 1))[:, 1]

    titulo_curva_calib = 'Curva de probabilidad calibrada (val. cruzada)' if nombre_subgrupo=='' else f'Curva de probabilidad calibrada (val. curzada) para {nombre_subgrupo}'
    log_calibration_analysis(y_score=scores_eval_cal, y_true=y_eval,  name=titulo_curva_calib, experiment=cmt_exp)


    if cmt_exp:

        fig_cm.canvas.draw()
        fig_pr.canvas.draw()
        fig_roc.canvas.draw()

        cmt_exp.log_metric(f'prec@{threshold}' if nombre_subgrupo=='' else f'prec@{threshold}_' + nombre_subgrupo, precision)
        cmt_exp.log_metric(f'acc@{threshold}' if nombre_subgrupo=='' else f'acc@{threshold}_' + nombre_subgrupo, accuracy)
        cmt_exp.log_metric(f'recall@{threshold}' if nombre_subgrupo==''  else f'recall@{threshold}_' + nombre_subgrupo, recall)
        cmt_exp.log_metric(f'AP' if nombre_subgrupo=='' else f'AP_' + nombre_subgrupo, ap)
        cmt_exp.log_metric(f'roc_auc' if nombre_subgrupo=='' else f'roc_auc_' + nombre_subgrupo, roc_auc)
        cmt_exp.log_figure('confusion_matrix' if nombre_subgrupo=='' else f'confusion_matrix_' + nombre_subgrupo,figure=fig_cm)
        cmt_exp.log_figure('precision recall curve' if nombre_subgrupo=='' else f'precision recall curve_' + nombre_subgrupo,figure=fig_pr)
        cmt_exp.log_figure('ROC curve' if nombre_subgrupo=='' else f'ROC curve_' + nombre_subgrupo,figure=fig_roc)



def filtrar_datos(data, filtros):

    mask = pd.Series(True, index=data.index)
    
    for filt in filtros:
        col = filt["atributo"]
        op = filt["op"]
        value = filt.get("valor")
        
        print('Se eliminan las filas que satisfacen: ', col, op, value)
        mask &= operacion[op](data[col], value)

    data = data[~mask]
    
    return data

def run_experiment(args):


    #path = args.path
    
    trained_classifiers = []
    trained_classifiers_scores = []
    combination_criteria = ['hard_voting', 'soft_voting', 'weighted_soft_voting']
    for model_type in args.classifiers:
       
        cometExperiment = create_experiment(model_type)   if args.log_comet else None


        # ---------------- LOAD CONFIG ----------------
        config = load_config(args.config)
        filepath = config["data"].get('filepath')
        entrega = config["data"].get('entrega')
        print('Entrega:', entrega)
        seed = config["experiment"].get('seed')
        np.random.seed(seed)

        use_class_weights = (not args.disable_class_weights)
        use_sample_weights = (not args.disable_sample_weights)

        if use_class_weights:  
            if model_type=='XGBoost':
                num_pos = np.sum(y==1)
                num_neg = len(y) - num_pos
                scale_pos_weight = num_neg/num_pos
                class_weights = scale_pos_weight
            else:
                class_weights = 'balanced'        
        else :
            class_weights = None

        # Paso 1: cargo los datos
        #filename = 'IAE_procesada_2a_entrega.csv'  
        data = pd.read_csv(filepath)
        print('Dimension de los datos levantados:', data.shape)

        # Filtro algunos datos del conjunto entregado que no se van a usar para entrenar
        if config["filtros"]:
            data = filtrar_datos(data, config['filtros'])
        print('Dimension de los datos luego de filtrar:', data.shape)


        # leo del archivo de configuración los atributos a usar 
        num_attribs = config["data"].get('num_features')
        cat_attribs = config["data"].get('cat_features')
        atributos = {'numericos':num_attribs, 'categoricos': cat_attribs}
        cv_split = config['data'].get('cv_split')  
        # obtengo los features que se usan para entrenar
        X = getX(data, num_attribs, cat_attribs)
        #y = getY(data, metodo=args.y_method)
        # obtengo el target
        target = config['data'].get('target') 
        y = data[target] ==1
        print(X.info())
        print(X.shape, y.shape)

        # ---------------- COMET ----------------
        if args.log_comet:
            cometExperiment.log_parameters({'entrega': entrega})
            cometExperiment.add_tag(f'e{entrega}')
            cometExperiment.log_parameters(atributos)
            cometExperiment.log_dataset_hash(np.hstack((X.values,y.values[:,np.newaxis])))
            cometExperiment.add_tag(target)
            cometExperiment.add_tag(model_type)    
            cometExperiment.add_tag(args.gs_criteria)
            cometExperiment.add_tag(cv_split)
            if use_class_weights:
                cometExperiment.add_tag('class_weights')    

        plt = mostrar_histogramas(X.loc[:,num_attribs],y,num_attribs)
        if args.log_comet:
            cometExperiment.log_figure('Numéricas', plt)

        if cat_attribs:
            plt = mostrar_histogramas(X.loc[:,cat_attribs],y,cat_attribs)
            if args.log_comet:
                cometExperiment.log_figure('Categóricas', plt)

        
        folds = create_folds(cv_split)

        if 'grupo' in cv_split:
            grupo = data["CEDULA"]
            print('Folds generados utilizando grupos')
            for fold, (train_idx, test_idx) in enumerate(folds.split(X, y, grupo), start=1):
                print(f"Fold {fold}")
                print("Train:", train_idx)
                print("Test :", test_idx)
                num_positivos_fold_i = np.sum(y.values[test_idx]==1)
                num_negativos_fold_i = np.sum(y.values[test_idx]==0)
                print(f'positivos: {num_positivos_fold_i}, negativos: {num_negativos_fold_i}')
                print()
        
        else:
            grupo = None
            print('Folds generados sin utilizar grupos')
            for fold, (train_idx, test_idx) in enumerate(folds.split(X,y), start=1):
                print(f"Fold {fold}")
                print("Train:", train_idx)
                print("Test :", test_idx)
                num_positivos_fold_i = np.sum(y.values[test_idx]==1)
                num_negativos_fold_i = np.sum(y.values[test_idx]==0)
                print(f'positivos: {num_positivos_fold_i}, negativos: {num_negativos_fold_i}')
                print()



    
        preprocessing = generate_preprocessing_pipeline(num_attribs,cat_attribs, classifier=model_type)
        

        gs_pipeline = build_model_with_cv(preprocessing, classifier=model_type, class_weights=class_weights,
                                        criteria=args.gs_criteria, random_state=seed, cv=folds)
        
        print('Cantidad de nans en el entrenamiento: ', X.isna().sum())

        if 'grupo' in cv_split:
            print("Grid Search considerando grupo PERSONA started")
            gs_pipeline.fit(X, y.values, groups=grupo)      
        else:
            print("Grid Search started")
            gs_pipeline.fit(X, y.values)


        #scores_oof = oof_prediction(X,y, gs_pipeline.best_estimator_)

        preprocess = gs_pipeline.best_estimator_.named_steps["preprocessing"]
        feature_names = preprocess.get_feature_names_out()

        if args.tsne:
            plt = mostrar_tsne(preprocess.transform(X),y)
            if args.log_comet:
                cometExperiment.log_figure('tsne', figure=plt)

        print("Mejores parámetros encontrados:")
        print(gs_pipeline.best_params_)

        print("\nMejor estimador:")
        print(gs_pipeline.best_estimator_)
        trained_classifiers.append(gs_pipeline.best_estimator_)

        plt.figure()
        result = permutation_importance(gs_pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)

        if cat_attribs:
            perm_importance = pd.Series(result.importances_mean, index=num_attribs+cat_attribs)
        else:
            perm_importance = pd.Series(result.importances_mean, index=num_attribs)

        perm_importance.sort_values().plot(kind="barh", figsize=(8,6))
        plt.title("Permutation Importance")
        #plt.show()
        if args.log_comet:
            cometExperiment.log_parameters(gs_pipeline.best_params_)
            cometExperiment.log_metric("best score", gs_pipeline.best_score_)
            cometExperiment.log_figure('Permutation importance', figure=plt)

        
        
        if model_type=='DecisionTree':
            plt.figure(figsize=(18,10))
            plot_tree(gs_pipeline.best_estimator_.named_steps["classifier"],fontsize=10,
                    filled=True, proportion=True, feature_names=feature_names)
            if args.log_comet: 
                cometExperiment.log_figure('Tree', figure=plt)
            
            plt.figure(figsize=(18,10))
            plot_tree(gs_pipeline.best_estimator_.named_steps["classifier"],
                    filled=True, max_depth=2,fontsize=10,
                    proportion=True, feature_names=feature_names)
            if args.log_comet:
                cometExperiment.log_figure('Tree (2 primeras ramas)', figure=plt)
            
        elif model_type=='RandomForest':
            rf = gs_pipeline.best_estimator_.named_steps["classifier"]
            importances = rf.feature_importances_
            feat_importance = pd.Series(importances, index=feature_names)
            feat_importance = feat_importance.sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            feat_importance.plot(kind='bar')
            plt.title("Importancia de Features - Random Forest")
            plt.ylabel("Importancia")
            plt.tight_layout()
            #plt.show()
            if args.log_comet:
                cometExperiment.log_figure('Feature importances', figure=plt)

        elif model_type=='XGBoost':
            plt_xg = plot_importance(gs_pipeline.best_estimator_.named_steps["classifier"], 
                                    importance_type='weight')
            if args.log_comet:
                cometExperiment.log_figure('Features importance (weight)', figure = plt_xg)
            plt_xg = plot_importance(gs_pipeline.best_estimator_.named_steps["classifier"], 
                                    importance_type='gain')
            if args.log_comet:
                cometExperiment.log_figure('Features importance (gain)', figure = plt_xg)

        elif model_type=='LogisticRegression':
            odds_ratios = np.exp(gs_pipeline.best_estimator_.named_steps["classifier"].coef_[0])
            print(feature_names)
            print(odds_ratios)
            feat_importance = pd.Series(odds_ratios, index=feature_names)
            feat_importance = feat_importance.sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            feat_importance.plot(kind='bar')
            plt.title("Odds Ratio - Logistic Regression")
            plt.ylabel("Importancia")
            plt.tight_layout()
            #plt.show()
            if args.log_comet:
                cometExperiment.log_figure('Odds Ratio', figure=plt)



 
        #save_gs_results(cometExperiment, gs_pipeline)

        plt = show_best_cv_results(gs_pipeline, args.gs_criteria)
        if args.log_comet:
          cometExperiment.log_figure('best cross val results',figure=plt)

        print('Cross validation...')
        cv_scores = cross_val_predict(gs_pipeline.best_estimator_, X, y,
                                        cv=folds, groups=grupo, method='predict_proba')[:, 1]
        trained_classifiers_scores.append(cv_scores)
        
        reportar_validacion_cruzada(cv_scores,y,threshold=0.5, cmt_exp=cometExperiment)

        

        for cond in config["subgrupo_evaluacion"]:
            mask = pd.Series(True, index=data.index)
            col = cond["atributo"]
            op = cond["op"]
            value = cond.get("valor")
            mask = operacion[op](data[col], value)
            print('Evaluando en grupo que satisface: ', col, op, value)

            reportar_validacion_cruzada(cv_scores,y,threshold=0.5, cmt_exp=cometExperiment, 
                                        indices_subgrupo=mask, nombre_subgrupo=f'{col} {op} {value}' )
            

        # ## Métricas restringidas a los intentos
        # cv_scores_intentos = cv_scores[indices_intentos]
        # y_intentos = y[indices_intentos]
        # cv_scores_no_intentos = cv_scores[~indices_intentos]
        # y_no_intentos = y[~indices_intentos]

        # precision_intentos = precision_score(y_intentos, cv_scores_intentos > threshold)
        # accuracy_intentos = accuracy_score(y_intentos, cv_scores_intentos > threshold)
        # recall_intentos = recall_score(y_intentos, cv_scores_intentos > threshold)
        
        # cometExperiment.log_metric(f'prec@{threshold}_intentos', precision_intentos)
        # cometExperiment.log_metric(f'acc@{threshold}_intentos', accuracy_intentos)
        # cometExperiment.log_metric(f'recall@{threshold}_intentos', recall_intentos)
        
        # ap_intentos = average_precision_score(y_intentos, cv_scores_intentos)
        # roc_auc_intentos = roc_auc_score(y_intentos, cv_scores_intentos)
        # cometExperiment.log_metric(f'AP_intentos', ap_intentos)
        # cometExperiment.log_metric(f'roc_auc_intentos', roc_auc_intentos)
        
        # plt = show_cv_confusion_matrix(cv_scores_intentos, y_intentos)
        # cometExperiment.log_figure('confusion_matrix_intentos',figure=plt)
        # plt = show_cv_precision_recall(cv_scores_intentos, y_intentos)
        # cometExperiment.log_figure('precision recall curve_intentos',figure=plt)
        # plt = show_cv_roc(cv_scores_intentos,y_intentos)
        # cometExperiment.log_figure('ROC curve_intentos',figure=plt)

        # if len(y_no_intentos) >0:
        #     plt = show_cv_confusion_matrix(cv_scores_no_intentos, y_no_intentos)
        #     cometExperiment.log_figure('confusion_matrix_no_intentos',figure=plt)
        #     plt = show_cv_precision_recall(cv_scores_no_intentos, y_no_intentos)
        #     cometExperiment.log_figure('precision recall curve_no_intentos',figure=plt)
        #     plt = show_cv_roc(cv_scores_no_intentos,y_no_intentos)
        #     cometExperiment.log_figure('ROC curve_no_intentos',figure=plt)

    

        if args.log_comet:
            cometExperiment.end()

    
    if len(trained_classifiers)>1 and args.combine:
        n_classifiers = len(args.classifiers)
        print('combining classifiers: ', n_classifiers)
        n_scores = trained_classifiers_scores[0].shape[0]
        cv_scores_all = np.zeros((n_classifiers, n_scores))
        predictions_all = np.zeros_like(cv_scores_all, dtype=bool)
        weights_ap = np.zeros(n_classifiers)
        for i, clf in enumerate(args.classifiers):
            cv_scores_all[i] = trained_classifiers_scores[i]
            predictions_all[i] = trained_classifiers_scores[i]>0.5
            weights_ap[i] = average_precision_score(y, trained_classifiers_scores[i])

        for comb_criteria in combination_criteria:
            
            if args.log_comet: 
                cometExperiment = create_experiment(comb_criteria)  
                for clf in args.classifiers: 
                    cometExperiment.add_tag(clf)
                cometExperiment.add_tag(comb_criteria)
            
            if comb_criteria=='hard_voting':
                votes = np.sum(predictions_all, axis=0)
                #predictions_i = votes > n_classifiers/2
                cv_scores_i = votes / n_classifiers
            elif comb_criteria=='soft_voting':
                cv_scores_i = np.mean(cv_scores_all,axis=0)
            elif comb_criteria=='weighted_soft_voting':
                cv_scores_i = np.average(cv_scores_all,axis=0,weights=weights_ap)
                #predictions_i = cv_scores_i>0.5
            
            
            reportar_validacion_cruzada(cv_scores_i,y,threshold=0.5, cmt_exp=cometExperiment)

            for cond in config["subgrupo_evaluacion"]:
                mask = pd.Series(True, index=data.index)
                col = cond["atributo"]
                op = cond["op"]
                value = cond.get("valor")
                mask = operacion[op](data[col], value)
                print('Evaluando en grupo que satisface: ', col, op, value)

                reportar_validacion_cruzada(cv_scores_i,y,threshold=0.5, cmt_exp=cometExperiment, 
                                            indices_subgrupo=mask, nombre_subgrupo=f'{col} {op} {value}' )
            
            if args.log_comet:
                cometExperiment.end()



def parseCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument( '--classifiers', nargs='+', choices=['RandomForest', 'XGBoost', 'LogisticRegression', 'DecisionTree'],
                                                    default=['RandomForest']
)
    parser.add_argument('--gs_criteria', type=str, default='roc_auc')
    parser.add_argument('--y_method', type=str, default='defuncion')
    parser.add_argument('--ams_threshold', type=float, default=0.9)
    parser.add_argument('--preprocessing', type=str, default='None')
    parser.add_argument('--split_factor', type=float, default=0.75)
    parser.add_argument('--disable_class_weights', '-dcw', action='store_true')
    parser.add_argument('--disable_sample_weights', '-dsw', action='store_true')
    parser.add_argument('--tsne', action='store_true')
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

    run_experiment(args)
