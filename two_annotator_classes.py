from utils import *
import numpy as np

from utils import *
from new_ideas import *
from multiple_reliability import obtain_posterior_multiple_rel

def second_delta(T:np.array, c:int=0):
    return T[c][c]/T[1-c][1-c]

def obtain_equal_rho(x:float, y:float, t:float):
    return 1 - ((1-x)*(1-y)*t)/(x*y + t -t*x -t*y)

def check_conditions_two_annotator_classes(T_a:np.array, T_b:np.array,
                                           vu:np.array, A_card:int,
                                           c:int=0, H:int=3):
    """
    Condition for annotators with 2 reliability classes from Section 3.4 of the main paper.
    """
    vu_ratio = vu[1-c]/vu[c]
    common_term = (second_delta(T_b, c)/second_delta(T_b, 1-c))**(H/2) 
    zita = (second_delta(T_b, 1-c)/second_delta(T_a, 1-c)) ** A_card
    left_side = (1/(np.sqrt(my_rho(T_b)))) * common_term * zita
    right_side = np.sqrt(my_rho(T_b)) * common_term * zita
    return (left_side < vu_ratio) and (right_side > vu_ratio)

def generate_T_matrices():
    """
    Generation of two classes of matrices so that rho_a = rho_b.
    """
    T_a_matrices = np.array([np.array([[0.58, 0.42], [.2, .8]]), 
                             np.array([[.78 ,.22], [.35,.65]])])
    
    T_b_matrices = np.array([np.array([[0.8, 0.2], [.42, .58]]), 
                             np.array([[.65 ,.35], [.22,.78]])])

    return T_a_matrices, T_b_matrices


if __name__ == '__main__':
    H = 7
    A_card = 4
    C = 2
    N = 10000
    vu_values = np.array([x/10+.05 for x in range(5,10)])
    T_values_a, T_values_b = generate_T_matrices()
    seed = 42
    seed_everything(seed=seed)
    for index, (T_a, T_b) in enumerate(zip(T_values_a, T_values_b)):
        for vu in tqdm(vu_values):

            true_labels = generate_true_labels(C=2, N=N, D=np.array([vu,1-vu]))
            annotations_A  = generate_annotations(true_labels,T_a,
                                                H=A_card, obtain_list=True,
                                                check_conditions=False)  
            annotations_B = generate_annotations(true_labels,T_b,
                                                H=H-A_card, obtain_list=True,
                                                check_conditions=False)
            
            data = np.concatenate((annotations_A, annotations_B), axis = 1 )
            
            T_matrices = []
            for _ in range(A_card):
                T_matrices.append(T_a)
            for _ in range(H-A_card):
                T_matrices.append(T_b)
            T_matrices=np.array(T_matrices)

            theoretical_res = check_conditions_two_annotator_classes(T_a,T_b, np.array([vu,1-vu]),A_card,0,H)
            map_acc = accuracy(true_labels,obtain_posterior_multiple_rel(data,T_matrices,true_labels,N,np.array([vu,1-vu])))
            e2wl, w2el, label_set = to_LA(data)
            mv_output = obtain_competitor_results('MV', e2wl, w2el, label_set)
            mv_acc = accuracy(mv_output, true_labels)
            
            print('Theoretical conditions: ',  theoretical_res)
            print('MV Accuracy: ', mv_acc)
            print('MAP Accuracy: ', map_acc)
            print(T_a[0][0], T_a[1][1], T_b[0][0], T_b[1][1], vu)
            print('--------')