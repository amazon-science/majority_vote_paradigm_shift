import pandas as pd
import numpy as np
import tqdm as tqdm

from utils import *
from new_ideas import *

def analyze_results(experiment_name:str='results/results_multiple_rel.csv'):
    results = {}
    df = pd.read_csv(experiment_name)
    for exp_id in df['Exp Id'].unique():
        results[exp_id] = {} 
        omap_acc, mv_acc = [], []
        modified_df = df[df['Exp Id']==exp_id]
        omap_acc = np.array(modified_df['oMAP Accuracy'].tolist())
        mv_acc = np.array(modified_df['MV Accuracy'].tolist())
        results[exp_id]['oMAP mean'] = round(np.mean(omap_acc), 4)
        results[exp_id]['oMAP std'] = round(np.std(omap_acc), 4)
        results[exp_id]['MV mean'] = round(np.mean(mv_acc), 4)
        results[exp_id]['MV std'] = round(np.std(mv_acc), 4)
        results[exp_id]['Condition satisfied'] = modified_df['Equal'].unique()[0]
    return results


def compute_R_c(T:np.array, c:int=0):
    value = (2/T[1-c][1-c]) + (2/T[c][1-c]) + (1/T[c][c]) + (1/T[1-c][c])
    return 1/value

def compute_A_c(T:np.array, vu:np.array,
                c:int=0, H:int=3):
    num = np.log(vu[1-c]/vu[c]) + H*np.log(T[1-c][1-c]/(1-T[c][c]))
    return num/my_rho(T)

def compute_sigma_lim(T:np.array, vu:np.array,
                      H:int, c:int=0):
    """
    sigma from section 3.4 of the main paper is computed.
    """
    A_c = compute_A_c(T, vu, c, H)
    minimum = min(A_c - np.floor(A_c), 1 - A_c - np.floor(A_c))
    return (np.log(my_rho(T))/H) * compute_R_c(T, c=0) * minimum

def perturbate_T_matrix(T:np.array, sigma:float):
    return T + np.array([[-1, 1],[1,-1]])*np.random.uniform(-sigma, sigma)


def obtain_posterior_multiple_rel(annotations:np.array, all_T_matrices:np.array,
                                  true_labels:list, N:int, vu:np.array):
    """
    Implementation of conditions from Section 3.4 of the main paper.
    """
    post = np.zeros(N, dtype=int)
    for i in range(N):
        C_set = np.where(annotations[i]==true_labels[i])[0] 
        W_set = np.where(annotations[i]!=true_labels[i])[0]
        c = true_labels[i]
        sum_C_c = sum(np.log(all_T_matrices[j][c, c]) for j in C_set)
        sum_W_c = sum(np.log(all_T_matrices[j][c, 1- c]) for j in W_set)
        sum_C_w = sum(np.log(all_T_matrices[j][1-c, c]) for j in C_set)
        sum_W_w = sum(np.log(all_T_matrices[j][1-c, 1-c]) for j in W_set)
        
        post_c = np.log(vu[c]) +  sum_C_c + sum_W_c 
        post_w = np.log(vu[1-c]) +  sum_C_w + sum_W_w
        if post_c> post_w:
            post[i] = c
        else: 
            post[i] = 1-c
    return post


if __name__ == '__main__':
    seed_everything(42)
    results = []
    T_s = np.array([np.array([[0.7, 0.3],[0.2,0.8]]), np.array([[0.55, 0.45],[0.45,0.55]]),
                  np.array([[0.9, 0.1],[0.3,0.7]]), np.array([[0.6, 0.4],[0.4,0.6]])])
    vu_s = np.array([np.array([0.6, 0.4]), np.array([0.5, 0.5]),
                   np.array([0.9, 0.1]), np.array([0.7, 0.3])])
    H = 3
    N = 10000
    exp_id = 0
    for T in tqdm(T_s):
        for vu in vu_s:
            true_labels = generate_true_labels(C=2, N=N, D=vu)
            for sim_number in range(6):
                sigma = min(compute_sigma_lim(T=T, vu=vu, H=H, c=0),
                            compute_sigma_lim(T=T, vu=vu, H=H, c=1))              
                all_T_matrices = []
                annotations = np.zeros((N,H), dtype=int)

                for h in range(H):
                    T_annotator = perturbate_T_matrix(T,sigma)
                    all_T_matrices.append(T_annotator)
                    annotations[:,h] =  [s[0] for s in generate_annotations(true_labels,
                                                                T_annotator, H=1, obtain_list=True,
                                                                check_conditions=False)]
                
                map_output = obtain_posterior_multiple_rel(annotations=annotations, all_T_matrices=all_T_matrices,
                                                    true_labels=true_labels, N=N, vu=vu)
                e2wl, w2el, label_set = to_LA(annotations)
                mv_output = obtain_competitor_results('MV', e2wl, w2el, label_set)
                mv_acc = accuracy(mv_output, true_labels)
                map_acc = accuracy(map_output, true_labels)
                results.append({'Exp Id' : exp_id,
                                'N' : N,
                                'T real' : T,
                                'vu real' : vu,
                                'Sim number' : sim_number,
                                'MV Accuracy' : mv_acc,
                                'oMAP Accuracy' : map_acc,
                                'Equal' : mv_acc == map_acc
                                })
            exp_id +=1
    df = pd.DataFrame(results)
    df = df.round(decimals=4)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/results_multiple_rel.csv', index=False)
    results = analyze_results()