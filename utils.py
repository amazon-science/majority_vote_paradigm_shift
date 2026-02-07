import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from crowdkit.aggregation import DawidSkene, MACE, GLAD, MajorityVote


from competitors.data_handling_competitors import gete2wlandw2el
from competitors.iwmv import iwmv
from competitors.mv import mv
from competitors.bwa import bwa
from competitors.la_method import *
from competitors.ebcc import ebcc_vb
from competitors.data_handling_competitors import seed_everything

def obtain_hatches(number_of_hatches):
    list_of_hatches = ['O', '.', '*', 'x']
    return list_of_hatches[:number_of_hatches]

def obtain_markers(number_of_markers):
    list_of_markers = ['o', 'v', 'p', '*', 'P', 's', 'X', '+', 'D', 'x']
    return list_of_markers[:number_of_markers]

def obtain_color(number_of_colors):
    all = ['#e41a1c', '#377eb8','#ff7f00','#4daf4a','#a65628',
        '#999999','#dede00','#f781bf','#984ea3']
    return all[:number_of_colors]

def my_rho(T):
    return (T[1][1]*T[0][0])/((1-T[0][0])*(1-T[1][1]))

def my_delta(Tc0,Tc1):
    return Tc0/(1-Tc1)


def second_delta(T:np.array, c:int=0):
    return T[c][c]/T[1-c][1-c]

def check_conditions_mv(T:np.array, vu:np.array, H:int=3,
                        c:int=0):
    """
    Given the T matrix and the ditribution of classes returns:
    True if the condission expressed in Theorem 3.4 from the paper is satisfied
    False otherwise
    This condition is valid for both the 1-coin case and the 2-coin case
    """
    ratio = vu[1-c]/vu[c]
    equal_term= (second_delta(T,c)/ second_delta(T,1-c))**(H/2)
    left_side = equal_term * (1/np.sqrt(my_rho(T)))
    right_side = equal_term * np.sqrt(my_rho(T))
    return (left_side < ratio) and (right_side > ratio)
    

def generate_table_with_std(data:dict):
    """
    Method which given the dict containing the results based on different seed values,
    computes the mean and the standard deviation
    """
    results = []
    for dataset, methods in data[list(data.keys())[0]].items():
        for method in methods.keys():
            values = [data[seed][dataset][method]['Result'] for seed in data.keys()]
            avg = np.mean(values)
            std = np.std(values)
            results.append({
                'Dataset': dataset,
                'Method': method,
                'Average': round(avg,4),
                'Std Dev': round(std,4),
            })

    df = pd.DataFrame(results)
    df = df.sort_values(['Dataset', 'Method'])
    return df

def get_synthetic_table_with_std(data:dict):
    """
    Same as the previous method but with synthetic results.
    """
    rows = []

    for seed, seed_data in data.items():
        for t_index, t_data in seed_data.items():
            for prob, prob_data in t_data.items():
                for method, value in prob_data.items():
                    rows.append([t_index, prob, method, value])

    df = pd.DataFrame(rows, columns=['T_index', 'Prob_value', 'Method', 'Value'])

    result = df.groupby(['T_index', 'Prob_value', 'Method']).agg({
        'Value': ['mean', 'std']
    }).reset_index()

    result.columns = ['T index', 'vu value', 'Method', 'Average', 'Std Dev']
    result = result.round(4)

    return result


def count_number_occurrences(list_a, list_b, list_c):
    """
    Used for the histogram.
    Count how many times the label aggregated by oracle MAP is equal to the one
    aggregated by MV and equal to the real label.
    """
    total = 0
    for a, b, c in zip(list_a, list_b, list_c):
        if a==b: 
            total +=1
    return total/len(list_a)


def obtain_toloka_method(method_name:str):
    if method_name == 'Majority Vote':
        return MajorityVote()
    elif method_name == 'Dawid-Skene':
        return DawidSkene()
    elif method_name == 'MACE':
        return MACE()
    elif method_name == 'GLAD':
        return GLAD()
    else:
        raise ValueError("Not supported method. The available are: GLAD, MACE, Majority Vote, Dawid-Skene.")

def obtain_competitor_results(method_name, e2wl, w2el, label_set, binary=False,
                              empirical_prior=False):
    """
    Results are transformed in a format which can then be used by
    the accuracy method.
    """
    if method_name.lower() == 'iwmv':
        truths, _ = iwmv(e2wl, w2el, label_set)
    elif method_name.lower() == 'la_one_pass':
        truths, _ = one_pass(e2wl, w2el, label_set, alpha=2, beta=2)
    elif method_name.lower() == 'la_two_pass':
        truths, a = one_pass(e2wl, w2el, label_set, alpha=2, beta=2)
        truths = two_pass(e2wl, a, label_set)
    elif method_name.lower() == 'bwa':
        truths, _, _ = bwa(e2wl, w2el, label_set,T_required=False)
        if binary:
            truths, _, _ = bwa(e2wl, w2el, label_set)
    elif method_name.lower() == 'ebcc':
        truths,_, _ = ebcc_vb(e2wl,w2el,label_set, empirical_prior=empirical_prior)
    elif method_name.lower() == 'mv':
        truths, _ = mv(e2wl, label_set)
    else:
        raise Exception(f'Method {method_name} not recognized.')
    result = list(truths.values())
    result = [int(x) for x in result]
    return result


def flatten_synthetic_results(input_dict, equality_precentage=True):
    """
    Method to flatten the results in a readable dataframe.
    """
    flattened_results = {}
    for T_index, first_dict in input_dict.items():
        for vu, second_dict in first_dict.items():
            for method, metrics in second_dict.items():
                flattened_results[(T_index, vu, method)] = {
                    'Result': metrics.get('Result', np.nan),
                    'Stat Sig': metrics.get('Stat Sig', np.nan),
                    'T-value': metrics.get('T value', np.nan),
                }
    result_df = pd.DataFrame.from_dict(flattened_results, orient='index')
    result_df = result_df.reset_index()
    result_df.columns = ['T index', 'vu', 'Method', 'Result', 'P-value', 'T-value']
    return result_df
    
def flatten_results(input_dict, equality_precentage=True):
    """
    Method to flatten the results in a readable dataframe.
    """
    flattened_results = {}
    for dataset, results in input_dict.items():
        for method, metrics in results.items():
            if equality_precentage:
                flattened_results[(dataset, method)] = {
                    'Result': metrics.get('Result', np.nan),
                    'Stat Sig': metrics.get('Stat Sig', np.nan),
                    'T-value': metrics.get('T value', np.nan),
                    'Equality Percentage': metrics.get('Equality Percentage', np.nan),
                }
            else:
                flattened_results[(dataset, method)] = {
                    'Result': metrics.get('Result', np.nan),
                    'Stat Sig': metrics.get('Stat Sig', np.nan),
                    'T-value': metrics.get('T value', np.nan),
                }
    result_df = pd.DataFrame.from_dict(flattened_results, orient='index')
    result_df = result_df.reset_index()
    if equality_precentage:
        result_df.columns = ['Dataset', 'Method', 'Result', 'P-value', 'T-value', 'Equality Percentage']
    else:
        result_df.columns = ['Dataset', 'Method', 'Result', 'P-value', 'T-value']
    return result_df

def obtain_annotations(input_df):
    """
    Transform the DataFrame input_df which takes as input in a dict
    with structure {sample ; [annotations], ....}
    """
    labels = {}
    for _, raw in input_df.iterrows():
        try:
            labels[raw['item']].append(raw['label'])
        except KeyError:
            labels[raw['item']] = [(raw['label'])]
    return labels


    
def majority_voting(input_df, need_labels=False, need_list:bool=False, num_classes:int=2):
    """
    Computes majority voting taking a DataFrame as input.
    It can outpu
    t a dict or an array
    """
    
    def compare_occurrences(arr):
        """Returns 0 if class 0 is the most probable, else 1.
        """
        count_0 = arr.count(0)
        count_1 = arr.count(1)

        if count_0 > count_1:
            return 0
        else:
            return 1
         
    if not need_labels:
        labels = obtain_annotations(input_df=input_df)
    else:
        labels = input_df
    if not need_list:
        count = {}
        for key,value in labels.items():
            count[key] = compare_occurrences(value)
    else:
        count = np.zeros(len(labels))
        for i in range(len(labels)):
            count[i] = np.argmax(np.bincount(labels[i], minlength=num_classes))
    return count


def compute_exact_predictions(gold_df, annotations):
    """
    Compute the accuracy wrt the gold labels.
    """
    final = 0
    if not isinstance(gold_df, list):
            true_labels = list(gold_df['truth'])
    else:
        true_labels = gold_df
    if isinstance(annotations, list):
        for true, noisy in zip(true_labels, annotations):
            if true == noisy:
                final +=1
    else:       
        for index, raw in gold_df.iterrows():
            if raw['truth'] == annotations[index]:
                final += 1
    return final/len(gold_df)


def compute_class_distribution(gold_df):
    """
    Simple method to compute the parameter vu of the paper (classes distribution)
    """
    num_classes = np.max(np.array(gold_df['truth'].tolist()))
    final = {str(i) : 0 for i in range(num_classes+1)}
    for _, raw in gold_df.iterrows():
        final[str(raw['truth'])] += 1
    distribution = np.empty(num_classes+1)
    for i in range(num_classes+1):
        distribution[i] = final[str(i)] / len(gold_df)
    return distribution

def annotation_dict_to_list(input_df):
    """
    Given annotations in a dict format, it returns the same
    annotations but in a list format."""
    if isinstance(input_df, dict):
        annotations = input_df
    else:
        annotations = obtain_annotations(input_df=input_df)
    min_length = 100
    for single_vector in annotations.values():
        if len(single_vector) < min_length:
            min_length = len(single_vector)
    annotations = list(annotations.values())
    return annotations

def oracle_MAP(Y, T, D, conditional=False, fixed_H=0, debug=False):
    """
    Oracle MAP method. Adapted to work with dicts.
    There is also a still variation to work with real data. 
    If the number of annotations in a dataset change, we can fix them to have a costant number of annotations.

    """
    N = len(Y)
    C = D.shape[0]
    map_Y = {}
    prior = np.log(D)
    if conditional:
        prior = np.zeros(C)  # ignore the prior and only compute the likelihood
    if fixed_H !=0:
        for index, _ in Y.items():
            Y[index] = Y[index][:fixed_H]
    for index in range(len(Y)): 
        log_posterior = prior + np.dot(np.log(T+1e-6), np.bincount(Y[index], minlength=C))
        map_Y[index] = np.argmax(log_posterior)
        if debug:
            print("Prior", prior)    
            print("Y index", Y[index])
            print("Bincount", np.bincount(Y[index], minlength=C))

    return map_Y

def create_list(list_length:int):
    output = [[] for _ in range(list_length)]
    return output

def generate_true_labels(C:int, N:int ,D):
    """
    Generation of labels given:
    C: number of classes
    N: number of samples in the dataset (size of the dataset)
    D: distribution of the classes
    Returns the real labels of the N total samples
    """
    true_Y = np.random.choice(C, N, p=D)
    return true_Y


def generate_annotations(true_Y, T, H:int, check_conditions:bool=True, obtain_list=True):
    """This method takes as input:
    true_Y : the samples from the dataset (and the related #samples)
    T: the T matrix
    H: #annotators
    And returns an array of arrays.
    The shape is:
        - number of samples (if dataset has 100 samples the array will have 100 samples)
        - number of annotators: if there are 7 annotators in there array each sample will be annotated 7 times

    """
    N = true_Y.shape[0]
    C = T.shape[0]

    if check_conditions:
        assert (T == T.transpose()).all(), "T matrix needs to be symmetric"
        assert (T.sum(axis=1) == np.ones(C)).all(), "T matrix needs to be stochastic"
    if obtain_list:
        noisy_Y = np.zeros((N, H), dtype=int)
    else:
        noisy_Y = {i: create_list(H) for i in range(1, N)}
    for i in range(N):
        noisy_Y[i] = np.random.choice(C, H, p=T[true_Y[i]]).tolist()
    return noisy_Y

def labels_to_list(input_data):
    result = []
    for item in input_data['truth']:
        result.append(item)
    return np.array(result)

def generate_real_T(y_true, y_noisy):
    """
    Anchor MAP method to estimate the Noise Transition Matrix.
    """
    num_classes = np.max(y_true) + 1
    noise_matrix = np.zeros((num_classes, num_classes))
    for t, n in zip(y_true, y_noisy):
        noise_matrix[t, n] += 1
    noise_matrix = noise_matrix / noise_matrix.sum(axis=1, keepdims=True)
    return noise_matrix

def compute_dataset_stats(input_df):
    num_classes = max(input_df['truth'])
    stats = np.zeros(num_classes+1)
    for item in input_df['truth']:
        stats[item] += 1
    return stats / len(input_df['truth']) * 100