from utils import *
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests 

from competitors.data_handling_competitors import gete2wlandw2el
from competitors.data_handling_competitors import seed_everything
from real_exps import parse_args
from iaa_api import InterAnnotatorAgreementAPI

def obtain_df(data):
    """
    Given a dictionary, returns a dataframe Pandas in a readable format.
    """
    records = []
    for key1, value1 in data.items():
        for key2, value2 in value1.items():
            record = {'key1': key1, 'key2': key2}
            record.update(value2)
            records.append(record)

    df = pd.DataFrame(records)

    df.columns = ['T value', 'vu value', 'Oracle MAP', 'Estimated MAP','MACE', 'LA_op', 'LA_tp', 'BWA'] #'MV', 'IWMV', 'Dawid-Skene', 'MACE', 'GLAD']
    return df

def accuracy(true_Y: list, aggregated: list):
    """
    Method to compute the accuracy
    """
    final = 0
    for true, noisy in zip(true_Y, aggregated):
        if true == noisy:
            final +=1
    return final/len(true_Y)

def to_toloka(input_data:list):
    """
    Transforms the annotation in a format list of lists to a dataframe readable by Toloka.
    """
    results = []
    for item, single_annotation in enumerate(input_data):
        for worker, label in enumerate(single_annotation):
            results.append({'task': item, 'worker': worker, 'label': label})
    return pd.DataFrame(results, index=None)

def to_LA(input_data:list):
    """
    Transforms the annotation in a format list of lists to dicts
    which can be used by competitors.
    """
    e2wl, w2el, label_set = gete2wlandw2el(None, input_data)
    return e2wl, w2el, label_set

def generate_exps(num_classes:int, num_samples:int, vu: float,
                  H:int, T:np.array, return_toloka:bool=False):
    """
    Given:
    H: number of annotators
    num_classes: number of classes
    num_samples: number of samples which need to be annotated
    vu: distribuution of the classes
    T: noise transition matrix
    Generates the required experiments."""
    true_labels = generate_true_labels(C=num_classes, N=num_samples, D=[vu, 1-vu])
    data = generate_annotations(true_labels, T, H=H, obtain_list=True,
                                check_conditions=False)
    if return_toloka:
        toloka = to_toloka(data)
        return data, true_labels, toloka
    else:
        return data, true_labels


if __name__ == '__main__':
    results = {}
    toloka_methods = ['Dawid-Skene', 'MACE', 'GLAD']
    other_methods = ['la_one_pass', 'la_two_pass', 'BWA', 'MV', 'IWMV']
    H = 3
    num_classes = 2
    num_samples = 10000
    vu_values = np.array([x/10 for x in range(1,10)])
    T_values = ([np.array([[0.8,0.2], [0.2,0.8]]), np.array([[0.51,0.49], [0.49,0.51]]),
                         np.array([[0.9,0.1], [0.1,0.9]]), np.array([[0.6,0.4], [0.4,0.6]]),
                         np.array([[0.7,0.3], [0.3,0.7]]),
                        np.array([[0.6,0.4], [0.25,0.75]]), np.array([[0.6,0.4], [0.1, 0.9]])])
    args = parse_args()
    if args.seed_values == [42]:
        all_p_values = {}
        seed = 42
        seed_everything(seed=seed)
        for index, T in enumerate(T_values):
            results[index] = {}
            for vu in tqdm(vu_values):
                results[index][vu] = {}
                data, true_labels, toloka_data = generate_exps(num_classes=num_classes, num_samples=num_samples, vu=vu,
                            H=H, T=T, return_toloka=True)
                
                oracle_results = list(oracle_MAP(data, T, np.array([vu, 1-vu])).values())
                results[index][vu]['Oracle MAP'] = {}
                results[index][vu]['Oracle MAP']['Result'] = accuracy(true_labels, oracle_results)

                iaa = InterAnnotatorAgreementAPI(data)
                iaa._build_t_matrix()
                estimated_map = list(oracle_MAP(data, iaa._t_hat, np.array(iaa._label_distribution)).values())
                results[index][vu]['Estimated MAP'] = {} 
                results[index][vu]['Estimated MAP']['Result'] = accuracy(true_labels, estimated_map)
                if estimated_map != oracle_results:
                    t_value, p_value = stats.wilcoxon(estimated_map, oracle_results)
                else:
                    t_value, p_value = -1, -1
                results[index][vu]['Estimated MAP']['T value'] = round(t_value, 6)
                all_p_values['Estimated MAP'] = p_value

                e2wl, w2el, label_set = to_LA(data)
                for method in other_methods:
                    result = obtain_competitor_results(method, e2wl, w2el, label_set, binary=True)
                    results[index][vu][method] = {}
                    results[index][vu][method]['Result'] = accuracy(true_labels, result)
                    if result != oracle_results:
                        t_value, p_value = stats.wilcoxon(result, oracle_results)
                    else:
                        t_value, p_value = -1, -1
                    results[index][vu][method]['T value'] = round(t_value, 6)
                    all_p_values[method] = round(p_value, 6)


                for method_name in toloka_methods:
                    method = obtain_toloka_method(method_name)
                    result = method.fit_predict(toloka_data)
                    results[index][vu][method_name] = {}
                    results[index][vu][method_name]['Result'] = accuracy(true_labels, result)
                    if result.values.tolist() != oracle_results:
                        t_value, p_value = stats.wilcoxon(result, oracle_results)
                    else:
                        t_value, p_value = -1, -1
                    results[index][vu][method_name]['T value'] = round(t_value, 6)
                    all_p_values[method_name] = round(p_value, 6)

                rejected , corrected_p_values, _, _ = multipletests(list(all_p_values.values()), alpha=0.05, method='bonferroni')
                for (name, value), reject  in zip(all_p_values.items(), rejected):
                    results[index][vu][name]['Stat Sig'] = reject
        df = flatten_synthetic_results(results)
        df = df.round(decimals=4)
        os.makedirs('results', exist_ok=True)
        df.to_csv(f'results/synthetic_results_H_{H}_N_{num_samples}.csv', index=False)
    else:
        for seed in args.seed_values:
            results[seed] = {}
            for index, T in enumerate(T_values):
                results[seed][index] = {}
                for vu in tqdm(vu_values):
                    results[seed][index][vu] = {}
                    data, true_labels, toloka_data = generate_exps(num_classes=num_classes, num_samples=num_samples, vu=vu,
                                H=H, T=T, return_toloka=True)
                    
                    oracle_results = list(oracle_MAP(data, T, np.array([vu, 1-vu])).values())
                    results[seed][index][vu]['Oracle MAP'] = accuracy(true_labels, oracle_results)

                    iaa = InterAnnotatorAgreementAPI(data)
                    iaa._build_t_matrix()
                    estimated_map = list(oracle_MAP(data, iaa._t_hat, np.array(iaa._label_distribution)).values())
                    results[seed][index][vu]['Estimated MAP'] = accuracy(true_labels, estimated_map)

                    e2wl, w2el, label_set = to_LA(data)
                    for method in other_methods:
                        result = obtain_competitor_results(method, e2wl, w2el, label_set, binary=True)
                        results[seed][index][vu][method] = accuracy(true_labels, result)

                    for method_name in toloka_methods:
                        method = obtain_toloka_method(method_name)
                        result = method.fit_predict(toloka_data)
                        results[seed][index][vu][method_name] = accuracy(true_labels, result)
        df = get_synthetic_table_with_std(results)
        df = df.round(decimals=4)
        os.makedirs('results', exist_ok=True)
        df.to_csv(f'results/synthetic_results_H_{H}_N_{num_samples}.csv')
                