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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier 

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

def getX(data, num_atributs, cat_attribs):
    # Este data frame (77 casos) incluye personas que tuvieron IAE y además murieron en 2023 (por suicidio u otras causas)
    
    if 'Sexo' in num_atributs:
        data['Sexo']=data['Sexo']=='Masculino'
    if 'IAE_PREVIO_CORREGIDO' in num_atributs:
        data['IAE_PREVIO_CORREGIDO']=data['IAE_PREVIO_CORREGIDO']=='SI'

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
    return y    



def generate_preprocessing_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
    ("standardize", StandardScaler()),
    ])

    cat_pipeline = make_pipeline(
        #SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))


    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
        ], remainder='passthrough')
    
    return preprocessing



def build_model_with_cv(preprocessing, classifier='LogisticRegression', class_weights=None,cv=5, criteria='roc_auc', n_jobs=-1):
    """
    Construye un pipeline con preprocesamiento y optimización de hiperparámetros
    mediante validación cruzada, para distintos clasificadores.
    """

    # Modelos base
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000,class_weight=class_weights),
        'RandomForest': RandomForestClassifier(class_weight=class_weights),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(probability=True,class_weight=class_weights),
        'DecisionTree': DecisionTreeClassifier(class_weight=class_weights),
        'HistGradientBoosting': HistGradientBoostingClassifier(class_weight=class_weights),
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
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 8],
            'classifier__learning_rate': [0.01, 0.1, 0.3]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        },
        'DecisionTree': {
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
    }

    # Verificación
    if classifier not in classifiers:
        raise ValueError(f"Clasificador '{classifier}' no soportado. Opciones: {list(classifiers.keys())}")

    # Construcción del pipeline
    base_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifiers[classifier])
    ])

    # Configurar búsqueda con CV
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grids[classifier],
        cv=cv,
        n_jobs=n_jobs,
        scoring=criteria,
        return_train_score=False
    )

    return grid_search


def show_cross_validation_results(gs_pipeline):

    #for clase, scores in classifier.scores_.items():
    #    print(f"Clase {clase}: shape {scores.shape}")  # (n_folds, n_Cs)

    #scores = list(classifier.scores_.values())[0]  # si es binario, tomamos la clase positiva
    #mean_scores = np.mean(scores, axis=0)
    #std_scores = np.std(scores, axis=0)

    import matplotlib.pyplot as plt
    mean_scores = gs_pipeline.cv_results_['mean_test_score']
    std_scores = gs_pipeline.cv_results_['std_test_score']

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
    plt.ylabel("Mean Cross-Validation Score")
    plt.title("Cross-Validation Performance")
    plt.legend()
    plt.show()





def show_cv_confusion_matrix(cv_scores, y, threshold=0.5, filename=None):


    # Calcular la matriz de confusión
    cm = confusion_matrix(y, cv_scores>threshold)

    # Mostrarla
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Matriz de confusión (validación cruzada), th={threshold:.02f}')
    plt.savefig('CV_conf_matrix.png' if filename is None else filename)
    plt.show()

def show_cv_precision_recall(cv_scores, y, filename=None):
    
    precision, recall, _ = precision_recall_curve(y, cv_scores)
    ap = average_precision_score(y, cv_scores)

    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-validated Precision–Recall Curve")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.savefig('CV_PR.png' if filename is None else filename)
    plt.show()


def show_cv_roc(cv_scores, y, filename=None):

    #y_score = model1_pipeline.predict_proba(data1_X)[:, 1]  # probabilidad de la clase positiva
    fpr, tpr, _ = roc_curve(y, cv_scores)
    auc = roc_auc_score(y, cv_scores)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross Validated ROC curve")
    plt.legend()
    plt.savefig('Cross Validated ROC curve'  if filename is None else filename)
    plt.show()




filename = 'IAE_procesada.csv'
data = pd.read_csv(filename)
columnas = data.columns.to_list()


print(columnas)

num_attribs = ["GRUPO_EDAD_", 'Sexo',"IAE_PREVIO_CORREGIDO","NUMERO_INTENTOS_",
              'DIAS_PROMEDIO_INTENTOS_','PRESTADOR_PUBLICO_','PRESTADOR_PRIVADO_'] 
cat_attribs = ["DECISION_", "METODO_IAE_PREVIO_"]
X = getX(data, num_attribs, cat_attribs)
y = getY(data)


preprocessing = generate_preprocessing_pipeline(num_attribs,cat_attribs)
gs_pipeline = build_model_with_cv(preprocessing, classifier='RandomForest', class_weights='balanced',
                                  criteria='recall')
gs_pipeline.fit(X, y.values)

print("Mejores parámetros encontrados:")
print(gs_pipeline.best_params_)

print("\nMejor estimador:")
print(gs_pipeline.best_estimator_)

resultados = pd.DataFrame(gs_pipeline.cv_results_)
resultados = resultados.sort_values('mean_test_score', ascending=False)

# Mostrar columnas relevantes
print(resultados[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

print(gs_pipeline.cv_results_['std_test_score'])

show_cross_validation_results(gs_pipeline)

cv_scores = cross_val_predict(gs_pipeline, X, y,
                                cv=5, method='predict_proba')[:, 1]

show_cv_confusion_matrix(cv_scores, y)
show_cv_precision_recall(cv_scores, y)
show_cv_roc(cv_scores,y)