from comet_ml import Experiment
from matplotlib import pyplot as plt
import random
import os




def get_api_key():
    return 'zP7XMvXyKy18NM38ds0BQKMiv' #'AHsKk9TVlSTeoWIQN3G17fI3G'

def create_experiment(name):
    experiment_number = random.random()

    # create an experiment with your api key
    exp = Experiment(api_key=get_api_key(),  # api_key=os.environ.get("COMET_API_KEY"), #
                                 project_name='Solucionadores')
    exp.set_name(name)
    exp.log_parameter('Experiment Number', experiment_number)

    return exp

def save_metrics(cmt_exp, th,  precision, recall, f1score, ams):
    metrics = {
        'precision@%f'%th :precision,
        'recall@%f'%th: recall,
        'f1score@%f'%th: f1score,
        'ams@%f'%th:ams
    }

    cmt_exp.log_metrics(metrics)

def save_gs_results(cmt_exp, grid_search):
    metrics = {
        'mean_fit_time': grid_search.cv_results_['mean_fit_time'],
        'std_fit_time': grid_search.cv_results_['std_fit_time'],
        'best_test_score_mean': grid_search.cv_results_['mean_test_score'][grid_search.best_index_],
        'best_test_score_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
    }

    plt.figure()
    plt.plot(grid_search.cv_results_['mean_test_score'], '*-')
    plt.xlabel('fold')
    plt.ylabel('mean_test_score')
    plt.title('Cross validation score')
    plt.grid()

    cmt_exp.log_metrics(metrics)

    for i in range(len(grid_search.cv_results_['params'])):
        for k, v in grid_search.cv_results_.items():
            if k == "params":
                cmt_exp.log_parameters(v[i])
            else:
                cmt_exp.log_metric(k, v[i])

    cmt_exp.log_figure(figure=plt)

def save_cv_results(cmt_exp, cross_val_results):
    '''
    Entrada:
        cmt_exp: experimento comet
        cross_val_scores: scikit-learn cross validation results
    '''
    avg_fit_time = cross_val_results['fit_time'].mean()
    std_fit_time = cross_val_results['fit_time'].std()
    avg_cv_accuracy = cross_val_results['test_accuracy'].mean()
    std_cv_accuracy = cross_val_results['test_accuracy'].std()
    avg_cv_precision = cross_val_results['test_precision'].mean()
    std_cv_precision = cross_val_results['test_precision'].std()
    avg_cv_recall = cross_val_results['test_recall'].mean()
    std_cv_recall = cross_val_results['test_recall'].std()

    metrics = {
        'avg_fit_time': avg_fit_time,
        'std_fit_time': std_fit_time,
        'avg_cv_accuracy': avg_cv_accuracy,
        'std_cv_accuracy': std_cv_accuracy,
        'avg_cv_precision': avg_cv_precision,
        'std_cv_precision': std_cv_precision,
        'avg_cv_recall': avg_cv_recall,
        'std_cv_recall': std_cv_recall,
    }

    plt.figure()
    plt.plot(cross_val_results['test_accuracy'], '*-')
    plt.xlabel('fold')
    plt.ylabel('accuracy')
    plt.title('Cross validation accuracy')
    plt.grid()

    cmt_exp.log_metrics(metrics)
    cmt_exp.log_figure(figure=plt)

    return