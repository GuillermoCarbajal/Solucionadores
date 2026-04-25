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
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier 

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.inspection import permutation_importance


from matplotlib import pyplot as plt

from sklearn.model_selection import KFold
from sklearn.base import clone

from utils.visualization import mostrar_tsne


import argparse

os.environ["COMET_AUTO_LOGGING"] = "0"

def getX(data, num_attribs, cat_attribs):
    # Este data frame (77 casos) incluye personas que tuvieron IAE y además murieron en 2023 (por suicidio u otras causas)
    
    if 'Sexo' in num_attribs:
        data['Sexo']=data['Sexo']=='Masculino'
    if 'IAE_PREVIO_CORREGIDO' in num_attribs:
        data['IAE_PREVIO_SI']=data['IAE_PREVIO_CORREGIDO']=='SI'
        data['IAE_PREVIO_NO']=data['IAE_PREVIO_CORREGIDO']=='NO'
        num_attribs.remove('IAE_PREVIO_CORREGIDO')
        num_attribs.append('IAE_PREVIO_SI')
        num_attribs.append('IAE_PREVIO_NO')
    if 'RUCAF_cobertura' in num_attribs:
        data['RUCAF_cobertura_fonasa']=data['RUCAF_cobertura']=='Fonasa'
        data['RUCAF_cobertura_no_fonasa']=data['RUCAF_cobertura']=='No Fonasa'
        num_attribs.remove('RUCAF_cobertura')
        num_attribs.append('RUCAF_cobertura_fonasa')
        num_attribs.append('RUCAF_cobertura_no_fonasa')
        

    X = data[num_attribs+cat_attribs]

    
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


    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
        ], remainder='passthrough')
    
    return preprocessing



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
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        },
        'RandomForest': {
            'classifier__n_estimators': [100, 200, 500, 1000],
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 500, 1000],
            'classifier__max_depth': [3, 5, 8],
            'classifier__learning_rate': [0.001, 0.01, 0.1, 0.3]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        },
        'DecisionTree': {
            'classifier__max_depth': [3, 5, 10, None],
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
        return_train_score=False, verbose=1
    )

    return grid_search


def show_cross_validation_results(gs_pipeline, criteria):

    #for clase, scores in classifier.scores_.items():
    #    print(f"Clase {clase}: shape {scores.shape}")  # (n_folds, n_Cs)

    #scores = list(classifier.scores_.values())[0]  # si es binario, tomamos la clase positiva
    #mean_scores = np.mean(scores, axis=0)
    #std_scores = np.std(scores, axis=0)

    import matplotlib.pyplot as plt
    mean_scores = gs_pipeline.cv_results_[f'mean_test_{criteria}']
    std_scores = gs_pipeline.cv_results_[f'std_test_{criteria}']

    plt.figure(figsize=(6,4))
    plt.semilogx( mean_scores, marker='o', label='Mean CV Score')
    plt.fill_between(
        np.arange(len(mean_scores)),
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2,
        label='±1 Std. Dev.'
    )
    #plt.xlabel("C (Inverse of Regularization Strength)")
    plt.ylabel(f"Mean Cross-Validation {criteria}")
    plt.title("Cross-Validation Performance")
    plt.legend()
    return plt
    #plt.show()





def show_cv_confusion_matrix(cv_scores, y, threshold=0.5, filename=None):


    # Calcular la matriz de confusión
    cm = confusion_matrix(y, cv_scores>threshold)

    # Mostrarla
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Matriz de confusión (validación cruzada), th={threshold:.02f}')
    #plt.savefig('CV_conf_matrix.png' if filename is None else filename)
    #plt.show()
    return plt

def show_cv_precision_recall(cv_scores, y, filename=None):
    
    precision, recall, _ = precision_recall_curve(y, cv_scores)
    ap = average_precision_score(y, cv_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-validated Precision–Recall Curve")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.savefig('CV_PR.png' if filename is None else filename)
    #plt.show()


def show_cv_roc(cv_scores, y, filename=None):

    #y_score = model1_pipeline.predict_proba(data1_X)[:, 1]  # probabilidad de la clase positiva
    fpr, tpr, _ = roc_curve(y, cv_scores)
    auc = roc_auc_score(y, cv_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross Validated ROC curve")
    plt.legend()
    plt.savefig('Cross Validated ROC curve'  if filename is None else filename)
    #plt.show()



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
                counts = datos.value_counts(normalize=True)

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


def run_experiment(args):

    cometExperiment = create_experiment(args.model_type)
    #path = args.path

    # ---------------- LOAD CONFIG ----------------
    config = load_config(args.config)
    filepath = config["data"].get('filepath')
    entrega = config["data"].get('entrega')
    print('Entrega:', entrega)
    seed = config["experiment"].get('seed')
    np.random.seed(seed)
    cometExperiment.log_parameters({'entrega': entrega})
    cometExperiment.add_tag(f'e{entrega}')


    # Paso 1: cargo los datos
    #filename = 'IAE_procesada_2a_entrega.csv'  
    data = pd.read_csv(filepath)
    columnas = data.columns.to_list()


    use_class_weights = (not args.disable_class_weights)
    use_sample_weights = (not args.disable_sample_weights)
 
    num_attribs = ["GRUPO_EDAD_", 'Sexo',"NUMERO_INTENTOS_"]
    num_attribs = config["data"].get('num_features')
    #            'DIAS_PROMEDIO_INTENTOS_','PRESTADOR_PUBLICO_','PRESTADOR_PRIVADO_'] 
    #            'PRESTADOR_PUBLICO_','PRESTADOR_PRIVADO_'] 
    
    #cat_attribs = ["DECISION_", "METODO_IAE_PREVIO_"]
    cat_attribs = ["METODO_IAE_PREVIO_","IAE_PREVIO_CORREGIDO"]
    cat_attribs = config["data"].get('cat_features')


    #data[cat_attribs]=data[cat_attribs].astype("category")
    #data['PRESTADOR_DIFF_']=data['PRESTADOR_PUBLICO_'].astype(float)-data['PRESTADOR_PRIVADO_'].astype(float)
    #num_attribs = num_attribs + ['PRESTADOR_DIFF_']

    atributos = {'numericos':num_attribs, 'categoricos': cat_attribs}  
    cometExperiment.log_parameters(atributos)

    X = getX(data, num_attribs, cat_attribs)
    #y = getY(data, metodo=args.y_method)
    target = config['data'].get('target') 
    y = data[target] ==1
    print(X.shape, y.shape)


    # Agrupar por todas las columnas de X y ver si y tiene más de un valor distinto
    df=X.copy()
    df['y']=y
    #conflictos = df.groupby(list(X.columns)).filter(lambda g: g["y"].nunique() > 1)

    #print(conflictos)
    #print(X.shape)
    #out = conflicting_patterns_counts(X, y, round_decimals=6)  # ajustá round_decimals si conviene
    #print("\nConflictos (patrón -> conteo por clase):")
    #print(out["conflicts"].head(50))


    duplicados = df.groupby(list(X.columns)).filter(lambda g: g["y"].nunique() > 1)

    indices_conflicto = duplicados.index.to_list()
    print('cantidad indices conflictos', len(indices_conflicto))

    # 1) Agrupamos por las columnas de X y contamos cuántas veces aparece cada clase
    pivot = (
        df.groupby(list(X.columns))["y"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # 2) Nos quedamos solo con los patrones conflictivos
    pivot_conflict = pivot[pivot.gt(0).sum(axis=1) > 1]

    print("Patrones conflictivos únicos:", len(pivot_conflict))
    print("Filas conflictivas totales:", pivot_conflict.sum().sum())

    pivot_conflict


    # identificar patrones conflictivos
    conflict_patterns = df.groupby(list(X.columns))["y"].nunique() > 1

    # extraer todos los patrones conflictivos
    keys_conflictivas = conflict_patterns[conflict_patterns].index

    # todas las filas cuyo patrón esté en un patrón conflictivo
    mask = df.apply(lambda row: tuple(row[list(X.columns)].values), axis=1).isin(keys_conflictivas)
    df_conflict_filas = df[mask]

    print("Filas conflictivas reales:", len(df_conflict_filas))

    

    cometExperiment.log_dataset_hash(np.hstack((X.values,y.values[:,np.newaxis])))
    num_pos = np.sum(y==1)
    num_neg = len(y) - num_pos
    scale_pos_weight = num_neg/num_pos
    cometExperiment.add_tag(target)

    print(X.info())

    if use_class_weights:  
        if args.model_type=='XGBoost':
            class_weights = scale_pos_weight
        else:
            class_weights = 'balanced'
        cometExperiment.add_tag('class_weights')
    else :
        class_weights = None

    cometExperiment.add_tag(args.model_type)    
    cometExperiment.add_tag(args.gs_criteria)

    print(columnas)

   
  

    plt = mostrar_histogramas(X.loc[:,num_attribs],y,num_attribs)
    cometExperiment.log_figure('Numéricas', plt)

    plt = mostrar_histogramas(X.loc[:,cat_attribs],y,cat_attribs)
    cometExperiment.log_figure('Categóricas', plt)

    preprocessing = generate_preprocessing_pipeline(num_attribs,cat_attribs, classifier=args.model_type)
    
    gs_pipeline = build_model_with_cv(preprocessing, classifier=args.model_type, class_weights=class_weights,
                                    criteria=args.gs_criteria, random_state=seed)
    

    print("Grid Search started")
    gs_pipeline.fit(X, y.values)

    preprocess = gs_pipeline.best_estimator_.named_steps["preprocessing"]
    feature_names = preprocess.get_feature_names_out()

    if args.tsne:
        plt = mostrar_tsne(preprocess.transform(X),y)
        cometExperiment.log_figure('tsne', figure=plt)

    print("Mejores parámetros encontrados:")
    print(gs_pipeline.best_params_)

    print("\nMejor estimador:")
    print(gs_pipeline.best_estimator_)

    
    if args.model_type=='DecisionTree':
        plt.figure()
        plot_tree(gs_pipeline.best_estimator_.named_steps["classifier"],proportion=True, feature_names=feature_names)
        cometExperiment.log_figure('Tree', figure=plt)
        
    elif args.model_type=='RandomForest':
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
        cometExperiment.log_figure('Feature importances', figure=plt)

        result = permutation_importance(gs_pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)

        perm_importance = pd.Series(result.importances_mean, index=num_attribs+cat_attribs)
        perm_importance.sort_values().plot(kind="barh", figsize=(8,6))
        plt.title("Permutation Importance")
        #plt.show()
        cometExperiment.log_figure('Permutation importance', figure=plt)

    elif args.model_type=='LogisticRegression':
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
        cometExperiment.log_figure('Odds Ratio', figure=plt)



    cometExperiment.log_parameters(gs_pipeline.best_params_)
    cometExperiment.log_metric("best score", gs_pipeline.best_score_)
    #save_gs_results(cometExperiment, gs_pipeline)

    plt = show_cross_validation_results(gs_pipeline, args.gs_criteria)
    cometExperiment.log_figure('cross val results',figure=plt)

    print('Cross validation...')
    cv_scores = cross_val_predict(gs_pipeline.best_estimator_, X, y,
                                    cv=5, method='predict_proba')[:, 1]
    threshold = 0.5
    precision = precision_score(y, cv_scores > threshold)
    accuracy = accuracy_score(y, cv_scores > threshold)
    recall = recall_score(y, cv_scores > threshold)
    ap = average_precision_score(y, cv_scores)
    roc_auc = roc_auc_score(y, cv_scores)

    cometExperiment.log_metric(f'prec@{threshold}', precision)
    cometExperiment.log_metric(f'acc@{threshold}', accuracy)
    cometExperiment.log_metric(f'recall@{threshold}', recall)
    cometExperiment.log_metric(f'AP', ap)
    cometExperiment.log_metric(f'roc_auc', roc_auc)
    
    #cv_scores = oof_predictions_with_best_params(gs_pipeline, X, y, cv=5)
    #save_cv_results(cometExperiment, cv_scores)
    plt = show_cv_confusion_matrix(cv_scores, y)
    cometExperiment.log_figure('confusion_matrix',figure=plt)
    plt = show_cv_precision_recall(cv_scores, y)
    cometExperiment.log_figure('precision recall curve',figure=plt)
    plt = show_cv_roc(cv_scores,y)
    cometExperiment.log_figure('ROC curve',figure=plt)

  
    # # Paso 2: crear el clasificador
    # classifier = createClassifier(args.model_type,create_comet = True)

    # # Paso 2: Preparación de los datos
    # X_train,y_train,X_val,y_val,nb_background_train,nb_signal_train,weights_train,weights_val = prepare_train_test_data(train,args.split_factor,classifier.getPreprocessingPipeline())
    # X_test = prepare_test_data(test,classifier.getPreprocessingPipeline())

    # # Paso 3: se setean los parametros
    # if use_class_weights:
    #     class_weights, scale_pos_weight = setClassWeightsParameters(nb_background_train,nb_signal_train)
    #     classifier.setClassWeights(class_weights,scale_pos_weight)
    
    # classifier.setTrainingParameters({'sample_weight':weights_train.values})


    # # Paso 4: entrenar el clasificador
    # cometExperiment.add_tag(args.gs_scoring)
    # gs_results = classifier.train(args.gs_scoring,X_train,y_train, cometExperiment, args.ams_threshold, use_sample_weights)
    # save_gs_results(cometExperiment, gs_results)

    # # Paso 5: salvar el clasificador a disco
    # models_dir="{}/models".format(args.working_dir)
    # if not os.path.exists(models_dir): os.mkdir(models_dir)
    # classifier.save(models_dir)

    # # Paso 6: validar los resultados en los datos de validación
    # print('Analyzing validation results...')
    # temp_path = "{}/temp_files".format(args.working_dir)
    # if not os.path.exists(temp_path): os.mkdir(temp_path)
    # classifier.validate(X_val,y_val,weights_val,temp_path, cometExperiment)
    # best_ams, best_th = classifier.getBestAMSandThresholds()
    # cometExperiment.log_metric('best ams val', best_ams)
    # cometExperiment.log_metric('best threshold val', best_th)

    cometExperiment.end()



def parseCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--model_type', type=str, default='RandomForest')
    parser.add_argument('--gs_criteria', type=str, default='roc_auc')
    parser.add_argument('--y_method', type=str, default='defuncion')
    parser.add_argument('--ams_threshold', type=float, default=0.9)
    parser.add_argument('--preprocessing', type=str, default='None')
    parser.add_argument('--split_factor', type=float, default=0.75)
    parser.add_argument('--disable_class_weights', '-dcw', action='store_true')
    parser.add_argument('--disable_sample_weights', '-dsw', action='store_true')
    parser.add_argument('--tsne', action='store_true')
 
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
