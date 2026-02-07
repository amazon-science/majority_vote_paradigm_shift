import time
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

from utils import *
from iaa_api import *
from syntethic_exps import *

def results_confirm_theory(entire_data:np.array, labels:np.array, T:np.array, vu:np.array):
    """
    We want to confirm what the theory is saying.
    To do so we aggregate using MAP or MV. If thir accuracy wrt. the real labels is equal we return True
    Othrewise we return a string containing the accuracy of both the approaches.
    """
    e2wl, w2el, label_set = to_LA(entire_data)
    majority_agg = obtain_competitor_results('MV', e2wl, w2el, label_set)
    majority_acc = accuracy(majority_agg, labels)

    map_agg = list(oracle_MAP(entire_data, T, vu).values())
    map_acc = accuracy(labels, map_agg)
    if map_acc == majority_acc:
        return True
    else:
        return f'They are not equal. MAP accuracy is {map_acc}, while MV accuracy is {majority_acc}.'

    
def use_mv_or_not(entire_data:np.array, percentage_for_estimation:float = 0.1,
                  estimation_method:str='iaa', estimation_required:bool=False,
                  debug:bool=False, H:int=3):
    start = time.time()
    data_for_estimation = entire_data[:int(percentage_for_estimation*len(entire_data))]
    api = InterAnnotatorAgreementAPI(data_for_estimation)
    if estimation_method == 'iaa':
        api._build_t_matrix()
        T_estimated = api.t_hat
        vu_estimated = api.label_distribution
    if estimation_method == 'ebcc':
        e2wl, w2el, label_set = to_LA(data_for_estimation)
        _, _, estimated_label_distribution, T_matrix = ebcc_vb(e2wl, w2el, label_set, T_required=True)
        vu_estimated = estimated_label_distribution
        T_estimated = T_matrix
    if estimation_method == 'iwmv':
        e2wl, w2el, label_set = to_LA(data_for_estimation)
        _, _, estimated_label_distribution, T_matrix = iwmv(e2wl, w2el, label_set, T_required=True)
        vu_estimated = estimated_label_distribution
        T_estimated = T_matrix

    if debug:
        print(f'Label distribution estimated: {vu_estimated}')
        print(f'T estimated: {T_estimated}')
    if not estimation_required:
        return check_conditions_mv(T=T_estimated, vu=vu_estimated, H=H), time.time() - start
    else:
        return check_conditions_mv(T=T_estimated, vu=vu_estimated, H=H), time.time() - start, vu_estimated, T_estimated

def what_is_the_best_a_priori(entire_data:np.array, toloka_data:np.array, percentage_for_estimation:float = 0.1,
                              true_labels:np.array=None, debug:bool=False):
    """
    The mathematical conditions is saying that MV is not as good as MAP.
    Since we can't use MAP, what method do we have to use?
    We try to predict it on the basis of a small subset of samples.
    """

    start = time.time()
    data_for_estimation = entire_data[:int(percentage_for_estimation*len(entire_data))]
    toloka_data_estimation = toloka_data[:int(percentage_for_estimation*len(toloka_data))]
    results = {}

    iaa = InterAnnotatorAgreementAPI(data_for_estimation)
    iaa._build_t_matrix()
    estimated_map_labels = list(oracle_MAP(data_for_estimation, iaa._t_hat, np.array(iaa._label_distribution)).values())
    
    toloka_methods = ['Dawid-Skene', 'MACE', 'GLAD']
    other_methods = ['la_one_pass', 'la_two_pass', 'BWA', 'IWMV', 'ebcc']

    e2wl, w2el, label_set = to_LA(data_for_estimation)
    for method in other_methods:
        result = obtain_competitor_results(method, e2wl, w2el, label_set, binary=True)
        results[method] = accuracy(estimated_map_labels, result)

    for method_name in toloka_methods:
        method = obtain_toloka_method(method_name)
        result = method.fit_predict(toloka_data_estimation)
        results[method_name] = accuracy(estimated_map_labels, result)
    
    if true_labels is not None:
        not_real_results = {}
        e2wl, w2el, label_set = to_LA(data)
        for method in other_methods:
            result = obtain_competitor_results(method, e2wl, w2el, label_set, binary=True)
            not_real_results[method] = accuracy(true_labels, result)

        for method_name in toloka_methods:
            method = obtain_toloka_method(method_name)
            result = method.fit_predict(toloka_data)
            not_real_results[method_name] = accuracy(true_labels, result)
        if debug:
            print(not_real_results)
        return f'The best method with our proposed approach is {max(results, key=results.get)} (accuracy {not_real_results[max(results, key=results.get)]}). Comparing with real gold labels the best method is {max(not_real_results, key=not_real_results.get)} (accuracy {not_real_results[max(not_real_results, key=not_real_results.get)]}). Time elapsed: {time.time() - start}'
    else:
        return f'The best method with our proposed approach is {max(results, key=results.get)} (accuracy {results[max(results, key=results.get)]}). Time elapsed: {time.time() - start}'

def run_competitors(entire_data:np.array, true_labels:np.array=None,
                    empirical_prior:bool=False, competitors:list=['ebcc', 'la_two_pass','MV']):
    competitors = [a for a in competitors if a != 'iaa']
    results = {}
    start = time.time() 
    e2wl, w2el, label_set = to_LA(entire_data)
    for method in competitors:
        start = time.time()
        if true_labels is not None:
            result = obtain_competitor_results(method, e2wl, w2el, label_set, binary=True,
                                               empirical_prior=empirical_prior)
            results[method] = (accuracy(true_labels, result), time.time() - start)
        else:
            results[method] = time.time() - start
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimation_methods', '-e', type=str, nargs='+', default=['ebcc','iaa', 'iwmv'], #ebcc takes looots of time!
                        help='Estimation methods')
    parser.add_argument('--num_samples', type=int, nargs='+', default=[5000, 50000], help='Number of samples')
    parser.add_argument('--nu_values', '-nu', type=float, nargs='+', default=[.9, .65, 0.7, .5, .6], help='nu values')
    parser.add_argument('--samples_percentage', '-sp', type=float, nargs='+', default=[.05, .1, .15], help='Samples percentage')
    return parser.parse_args()


if __name__ == '__main__':
    """
    We compute fraction of experiments where verification of Theorem 3.4 with estimated parameters from the candidate methods aligns with
    that of Theorem 3.4 using the true T and vu, considering cases where the theorem is verified with true parameters.
    """
    seed_everything(42)
    args = parse_args()
    debug = False
    H = 3
    T_values = np.array([np.array([[0.55,0.45], [0.2,0.8]]), np.array([[0.75,0.25], [0.35,0.65]]),
                         np.array([[0.65,0.35], [0.65,0.35]]), np.array([[0.7,0.3], [0.3,0.7]]),
                         np.array([[0.7,0.3], [0.45,0.55]])])
    C = 2
    exp_id = 0
    results = []

    for N in tqdm(args.num_samples):
        for T in T_values:
            for vu in args.nu_values:
                vu = np.array([vu, 1-vu])
                for sample_percentage in args.samples_percentage:
                    for estimation_method in args.estimation_methods:

                        data, true_labels, toloka_data = generate_exps(num_classes=C, num_samples=N, vu=vu[0],
                        H=H, T=T, return_toloka=True)
    
                        #I want to exctract a fraction of the data.
                        # Then I estimate vu and T and I try to see if I can keep using MV or not
                        if debug:
                            print(f'Label distribution real: {vu}')
                            print(f'T real: {T}')        

                        res_use_mv, time_required, vu_estimated, T_estimated = use_mv_or_not(data, sample_percentage, 
                                                                estimation_method=estimation_method,
                                                                estimation_required=True, H=H)
                        if debug:
                            print(f"The time required for the computation is: {time_required}")
                        res_confirm_theory = results_confirm_theory(data,true_labels, T, vu)
                        if debug:
                            if res_use_mv and res_confirm_theory:
                                print("Perfect")
                            elif not(res_use_mv) and res_confirm_theory is not True:
                                print("Well done!")
                                print(res_confirm_theory)
                            #print(what_is_the_best_a_priori(data, toloka_data, samples_perc_per_estimation, true_labels))
                            else:
                                print("Error in our estimation!")
                                print(res_confirm_theory)
                        empirical_prior = False
                        if res_use_mv:
                            empirical_prior = True
                        if debug:
                            print(run_competitors(entire_data=data, true_labels=true_labels, empirical_prior=empirical_prior))
                        current_values = run_competitors(entire_data=data, true_labels=true_labels,
                                                         empirical_prior=empirical_prior, competitors=['la_two_pass', 'ebcc', 'mv', 'iwmv'])
                        
                        omap_res = accuracy(true_labels, list(oracle_MAP(data, T, vu).values()))
                        for method, res in current_values.items():
                            results.append({'Exp Id': exp_id,
                                            'N' : N,
                                            'T real' : T,
                                            'T estimated' : T_estimated,
                                            'Estimation method' : estimation_method, 
                                            'Percentage for estimation' : sample_percentage,
                                            'vu real' : vu,
                                            'vu estimated' : vu_estimated,
                                            'Theorem satisfied real' : res_confirm_theory,
                                            'Theorem satisfied estimated' : res_use_mv,
                                            'Method' : method,
                                            'oMAP Accuracy' : omap_res,
                                            'Accuracy': res[0],
                                            'Time' : res[1],
                                            'Our Alg Time' : time_required
                                            })
                        exp_id+=1
    df = pd.DataFrame(results)
    df = df.round(decimals=4)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/results_estimation_exps.csv', index=False)