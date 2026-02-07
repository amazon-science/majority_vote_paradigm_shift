import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils import *
from syntethic_exps import *
from new_ideas import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, nargs='+', default=[5000, 10000, 20000, 50000, 65000, 100000],
                        help='Number of samples.')
    parser.add_argument('--nu_values', type=float, nargs='+', default=[.5], help='Nu values.')
    parser.add_argument('--samples_percentage', type=float, nargs='+', default=[.1], help='Percentage of samples.')
    parser.add_argument('--estimation_methods', type=str, nargs='+', default=['iaa', 'ebcc', 'iwmv'],
                        help='Estimation method.')
    return parser.parse_args()

def check_confusion_matrix_bound(filename: str = 'results/check_looseness_bound.csv',
                                 estimation_perc: float = 0.1,
                                 estimation_method: int = 'iaa',
                                 number_of_samples: int = 10000):
    """
    Compute the empirical and bound confusion matrices for a given set of parameters.

    Args:
        filename (str): The name of the CSV file containing the data.
        estimation_perc (float): The percentage used for estimation.
        estimation_method (int): The estimation method used.
        number_of_samples (int): The number of samples used.
    """
    df = pd.read_csv(filename)
    df['Oracle results'] = df['Oracle results'].apply(lambda x: True if str(x) == 'True' else False)


    df = df[(df['Percentage for estimation'] == estimation_perc) &
            (df['Estimation method'] == estimation_method) &
            (df['N'] == number_of_samples)
            ]
    bound_results = df['Bound results'].values
    empirical_results = df['Empirical results'].values
    oracle_results = df['Oracle results'].values
    array_empirical = confusion_matrix(oracle_results, empirical_results,
                                       normalize='all')
    array_bound = confusion_matrix(oracle_results, bound_results,
                                   normalize='all')
    return array_empirical, array_bound

def plot_for_all_N_values(filename:str='results/check_looseness_bound.csv',
                          estimation_perc:float=.1,
                          estimation_method:int='iaa'):
    all_n_samples = pd.read_csv(filename)['N'].unique().tolist()
    os.makedirs('results/confusion_matrices', exist_ok=True)
    for n_samples in all_n_samples:
        res = check_confusion_matrix_bound(filename=filename,
                                           estimation_perc=estimation_perc,
                                           estimation_method=estimation_method,
                                           number_of_samples=n_samples)
        disp = ConfusionMatrixDisplay(confusion_matrix=res[0])
        disp.plot()
        disp.ax_.set_title(f'Empirical, {n_samples} samples')
        disp.figure_.savefig(f'results/confusion_matrices/empirical_{n_samples}_{estimation_method}.png')
        disp = ConfusionMatrixDisplay(confusion_matrix=res[1])
        disp.plot()
        disp.ax_.set_title(f'Bound, {n_samples} samples')
        disp.figure_.savefig(f'results/confusion_matrices/bound_{n_samples}.png')

def plot_results(exp_name:str='results/check_looseness_bound.csv',
                 estimation_percentage:float=.1):
    """
    Plotting function
    """
    df = pd.read_csv(exp_name)
    df = df[df['Percentage for estimation'] == estimation_percentage]
    unique_vu_real = df['vu real'].unique()
    df['Oracle results'] = df['Oracle results'].apply(lambda x: True if str(x) == 'True' else x)

    for vu in unique_vu_real:
        group = df[df['vu real'] == vu]
        fig, ax = plt.subplots(figsize=(12, 6))
    
        N_values = sorted(group['N'].unique())
        bound_oracle_true = []
        bound_oracle_std = []
        for n in N_values:
            n_group = group[group['N'] == n]
            bound_oracle_true.append(np.mean(n_group['Bound results'] & n_group['Oracle results']))
            bound_oracle_std.append(np.std(n_group['Bound results'] & n_group['Oracle results']))
    
        methods = ['Bound & Oracle'] + list(group['Estimation method'].unique())
        colors = obtain_color(len(methods))
        markers = obtain_hatches(len(methods))
        num_methods = len(methods)
        x = np.arange(len(N_values))
        width = 0.8 / num_methods 

        ax.bar(x - (num_methods-1)*width/2, bound_oracle_true, width, yerr=bound_oracle_std, capsize=5,
            color=colors[0], label='Bound = Oracle', hatch=markers[0])
    
        for i, method in enumerate(group['Estimation method'].unique(), start=1):
            method_group = group[group['Estimation method'] == method]
            empirical_matches_oracle = []
            empirical_matches_oracle_std = []
            for n in N_values:
                n_method_group = method_group[method_group['N'] == n]
                empirical_matches_oracle.append(np.mean(n_method_group['Empirical results'] == n_method_group['Oracle results']))
                empirical_matches_oracle_std.append(np.std(n_method_group['Empirical results'] == n_method_group['Oracle results']))
            ax.bar(x - (num_methods-1)*width/2 + i*width, empirical_matches_oracle, width,
                yerr=empirical_matches_oracle_std, capsize=5, label=f'Empirical ({method.upper()}) = Oracle',
                color=colors[i], hatch=markers[i])
    
        ax.set_xlabel('N')
        ax.set_ylabel('TPR Estimated Conditions Th. 3.4')
        ax.set_title(r'$\nu=$' + f' {vu[1:4]}')
        ax.set_xticks(x)
        ax.set_xticklabels(N_values)
        ax.legend(loc='upper right')
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/bound_vs_empirical_{vu[1:4]}.pdf', format='pdf', dpi=600)
        plt.close()

"""
All these functions refer to Section 3.3 from the main paper.
When we can say by using estimated quantities that Theorem 3.4 is satisfied?
"""
def prob_function(gamma:float, epsilon:float,
                  N:int):
    return 1 - (2*gamma) + (2*np.exp(-2*(epsilon**2)*N))


def compute_eps_from_gamma(vu:np.array, C:int, T_tilde:np.array,
                           N:int, gamma:float):
    D = np.diag(vu)
    max_eig = np.max(np.linalg.eigvals(D))
    min_eig = np.min(np.linalg.eigvals(T_tilde))
    num = C * (np.sqrt(C)+1) * max_eig
    num /= min_eig
    product = np.sqrt((1/(2*N))*np.log((2*C**2)/gamma))
    return product * num

def compute_gamma_from_eps(vu:np.array, C:int, T_tilde:np.array,
                           N:int, epsilon:float):
    D = np.diag(vu)
    max_eig = np.max(np.linalg.eigvals(D))
    min_eig = np.min(np.linalg.eigvals(T_tilde))
    num = 2*C**2
    num_exp = epsilon**2 * min_eig**2 * (2*N)
    den_exp = C**2 * (np.sqrt(C)+1)**2 * max_eig**2
    return num / np.exp(num_exp/den_exp)

def my_rho(T):
    return (T[1][1]*T[0][0])/((1-T[0][0])*(1-T[1][1]))

def my_delta(Tc0,Tc1):
    return Tc0/(1-Tc1)

def compute_vu_tilde(T_tilde:np.array, vu:np.array):
    return np.linalg.inv(T_tilde).dot(vu)

def g_function(vu:float):
    return (1-vu) / vu

def h_function(T:np.array, H:int):
    ratio_0 = my_delta(T[0][0], T[1][1])/my_delta(T[1][1],T[0][0])
    return ratio_0**(H/2) *np.sqrt(my_rho(T))

def f_function(T:np.array, H:int):
    ratio_0 = my_delta(T[0][0], T[1][1])/my_delta(T[1][1],T[0][0])
    return ratio_0**(H/2) * 1 / (np.sqrt(my_rho(T)))

def check_conditions_estimated_data(vu_tilde:np.array, T_tilde:np.array,
                     H:int, num_classes:int=2,
                     eps:float=1e-2, eta:float=.05, xi:float=.05):
    if isinstance(vu_tilde, np.ndarray):
        vu_tilde = vu_tilde[0]
    diff = g_function(vu_tilde) - f_function(T_tilde, H)
    second_diff = h_function(T_tilde, H) - g_function(vu_tilde)
    min_eig = np.min(np.linalg.eigvals(T_tilde))
    psi = (eps/min_eig)*(1/(min_eig-eps) + np.sqrt(num_classes))*(1/(np.min(np.array([eta,1-eta])))**2)
    chi = (1-xi)**(-2) * 0.5 * np.sqrt(max(.5, (H+1)/2 - H*xi)**2 + max(.5, (H-1)/2 - H*xi)**2)
    first_cond = diff > psi + eps*chi
    second_cond = second_diff > psi + 4*eps*chi
    return first_cond and second_cond

def check_conditions_real_data(T:np.array, vu:np.array):
    return (f_function(T,H) < g_function(vu)) and (g_function(vu) < h_function(T,H))

def perform_exp(num_samples:np.array, T_values:np.array, nu_values:np.array, 
                samples_percentage:np.array, estimation_methods:np.array=np.array(['iaa']),
                H:int=3):
    """
    This method is able to generate Figure 4 from the main paper.
    We compute fraction of experiments where verification of Theorem 3.4 with estimated parameters from the candidate methods aligns with
    that of Theorem 3.4 using the true T and vu, considering cases where the theorem is verified with true parameters.
    We also show when Theorem 3.5 aligns with Theorem 3.4 using true parameters.
    Synthetic data have various sample sizes N.
    """
    num_classes = 2
    results = []

    for num_samples in tqdm(num_samples):
        for T in T_values:
            for vu in nu_values:
                vu = np.array([vu, 1-vu])
                for sample_percentage in samples_percentage:
                    for estimation_method in estimation_methods:

                        data, true_labels, _ = generate_exps(num_classes=num_classes, num_samples=num_samples, vu=vu[0],
                        H=H, T=T, return_toloka=True)

                        res_use_mv, _, vu_estimated, T_noisy = use_mv_or_not(data, sample_percentage, 
                                                                estimation_method=estimation_method,
                                                                estimation_required=True,H=H)

                        xi = 1 - np.max(np.array([T_noisy[0][0], T_noisy[1][1]]))
                        vu_tilde = compute_vu_tilde(T_noisy, vu_estimated)
                        eps = compute_eps_from_gamma(vu=vu_tilde, C=2, T_tilde=T_noisy, N=num_samples, gamma=.1)
                        eta = np.min(vu_tilde)
                        results_estimation_bound = check_conditions_estimated_data(vu_tilde[0], T_noisy, H, eps=eps,
                                                            xi=xi, eta=eta)
                        res_confirm_theory = results_confirm_theory(data,true_labels, T, vu)

                        results.append({    'N' : num_samples,
                                            'T real' : T,
                                            'T estimated' : T_noisy,
                                            'vu real' : vu,
                                            'vu estimated' : vu_estimated, 
                                            'Estimation method' : estimation_method, 
                                            'Percentage for estimation' : sample_percentage,
                                            'epsilon' : eps,
                                            'xi' : xi,
                                            'eta' : eta,
                                            'Bound results' : results_estimation_bound,
                                            'Empirical results' : res_use_mv,
                                            'Oracle results' : res_confirm_theory,
                                            })

    df = pd.DataFrame(results)
    df = df.round(decimals=4)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/check_looseness_bound.csv', index=False)

if __name__ == '__main__':
    args = parse_args()
    T_values = np.array([np.array([[0.6+i/100, 1-(0.6+i/100)], [1-(0.6+i/100),0.6+i/100]]) for i in range(5,40,5)])
    perform_exp(num_samples=args.num_samples, T_values=T_values, nu_values=args.nu_values,
                samples_percentage=args.samples_percentage, estimation_methods=args.estimation_methods)
    plot_results('results/check_looseness_bound.csv')
    plot_for_all_N_values('results/check_looseness_bound.csv',
                          estimation_method='iaa')
    
                        


