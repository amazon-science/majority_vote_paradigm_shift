from utils import *
from iaa_api import *
from scipy import stats
from statsmodels.stats.multitest import multipletests  
import argparse
from competitors.data_handling_competitors import seed_everything  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_values', '-s', type=int, nargs='+', default=[42], help='Values of the seed.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.seed_values == [42]:
        seed = 42
        seed_everything(seed=seed)
        toloka_methods = ['Dawid-Skene', 'MACE', 'GLAD']
        other_methods = ['ebcc','la_one_pass', 'la_two_pass', 'MV', 'IWMV', 'BWA']
        all_results = {}
        all_p_values = {}
        data_list  = sorted([entry for entry in os.listdir('data') if os.path.isdir(os.path.join('data', entry))])
        for item in tqdm(data_list):
            data = pd.read_csv(f'data/{item}/label.csv') 
            real_labels = pd.read_csv(f'data/{item}/truth.csv')
            dataset_stat = str(np.round(compute_dataset_stats(real_labels),2))
            item_mod = item + ' - ' + dataset_stat
            all_results[item_mod] = {}

            oracle_T = generate_real_T(labels_to_list(real_labels), annotation_dict_to_list(data))
            oracle_results = list(oracle_MAP(annotation_dict_to_list(data), oracle_T, compute_class_distribution(real_labels)).values())
            all_results[item_mod]['Oracle MAP'] = {} 
            all_results[item_mod]['Oracle MAP']['Result'] = round(compute_exact_predictions(real_labels, oracle_results), 3)

            modified_data = np.array(annotation_dict_to_list(data), dtype=object)
            iaa = InterAnnotatorAgreementAPI(modified_data)
            iaa._build_t_matrix()
            estimated_map = list(oracle_MAP(annotation_dict_to_list(data), iaa._t_hat, np.array(iaa._label_distribution)).values())
            all_results[item_mod]['Estimated MAP'] = {}
            all_results[item_mod]['Estimated MAP']['Result'] = round(compute_exact_predictions(real_labels, estimated_map), 3)
            if estimated_map != oracle_results:
                t_value, p_value = stats.wilcoxon(estimated_map, oracle_results)
            else:
                t_value, p_value = -1, -1
            all_results[item_mod]['Estimated MAP']['T value'] = round(t_value, 6)
            all_p_values['Estimated MAP'] = p_value

            if other_methods != []:
                e2wl, w2el, label_set = gete2wlandw2el(f'data/{item}/label.csv')
                for single_method in other_methods:
                    result = obtain_competitor_results(single_method, e2wl, w2el, label_set)
                    all_results[item_mod][single_method] = {}
                    all_results[item_mod][single_method]['Result'] = round(compute_exact_predictions(real_labels, result), 3)
                    t_value, p_value = stats.wilcoxon(result, oracle_results)
                    all_results[item_mod][single_method]['T value'] = round(t_value, 6)
                    all_p_values[single_method] = round(p_value, 6)
                    if single_method == 'MV':
                        all_results[item_mod][single_method]['Equality Percentage'] = count_number_occurrences(result, oracle_results, estimated_map)

            data = data.rename(columns={'item' : 'task'})

            if toloka_methods != [] : 
                for method_name in toloka_methods:
                    method = obtain_toloka_method(method_name)
                    result = method.fit_predict(data)
                    all_results[item_mod][method_name] = {}
                    all_results[item_mod][method_name]['Result'] = round(compute_exact_predictions(real_labels, result), 3)
                    t_value, p_value = stats.wilcoxon(result, oracle_results)
                    all_results[item_mod][method_name]['T value'] = round(t_value, 6)
                    all_p_values[method_name] = round(p_value, 6)

            rejected , corrected_p_values, _, _ = multipletests(list(all_p_values.values()), alpha=0.05, method='bonferroni')

            for (name, value), reject  in zip(all_p_values.items(), rejected):
                all_results[item_mod][name]['Stat Sig'] = reject

        result_df = flatten_results(all_results)
        result_df = result_df.round(decimals=4)
        os.makedirs('results', exist_ok=True)
        result_df.to_csv(f'results/real_data_results_{seed}.csv')
    else:
        all_results = {}
        toloka_methods = []#['Dawid-Skene', 'MACE', 'GLAD']
        other_methods = ['ebcc', 'BWA'] #['la_one_pass', 'la_two_pass', 'MV', 'IWMV', 'BWA']
        data_list  = sorted([entry for entry in os.listdir('data') if os.path.isdir(os.path.join('data', entry))])
        for seed in args.seed_values:
            seed_everything(seed=seed)
            all_results[seed] = {}
            for item in tqdm(data_list):
                print(f'Dataset: {item} - Seed: {seed}')
                data = pd.read_csv(f'data/{item}/label.csv') 
                real_labels = pd.read_csv(f'data/{item}/truth.csv')
                dataset_stat = str(np.round(compute_dataset_stats(real_labels),2))
                item_mod = item + ' - ' + dataset_stat
                all_results[seed][item_mod] = {}

                oracle_T = generate_real_T(labels_to_list(real_labels), annotation_dict_to_list(data))
                oracle_results = list(oracle_MAP(annotation_dict_to_list(data), oracle_T, compute_class_distribution(real_labels)).values())
                all_results[seed][item_mod]['Oracle MAP'] = {} 
                all_results[seed][item_mod]['Oracle MAP']['Result'] = round(compute_exact_predictions(real_labels, oracle_results), 3)

                modified_data = np.array(annotation_dict_to_list(data), dtype=object)
                iaa = InterAnnotatorAgreementAPI(modified_data)
                iaa._build_t_matrix()
                estimated_map = list(oracle_MAP(annotation_dict_to_list(data), iaa._t_hat, np.array(iaa._label_distribution)).values())
                all_results[seed][item_mod]['Estimated MAP'] = {}
                all_results[seed][item_mod]['Estimated MAP']['Result'] = round(compute_exact_predictions(real_labels, estimated_map), 3)

                if other_methods != []:
                    e2wl, w2el, label_set = gete2wlandw2el(f'data/{item}/label.csv')
                    for single_method in other_methods:
                        result = obtain_competitor_results(single_method, e2wl, w2el, label_set)
                        all_results[seed][item_mod][single_method] = {}
                        all_results[seed][item_mod][single_method]['Result'] = round(compute_exact_predictions(real_labels, result), 3)
                       
                data = data.rename(columns={'item' : 'task'})

                if toloka_methods != [] : 
                    for method_name in toloka_methods:
                        method = obtain_toloka_method(method_name)
                        result = method.fit_predict(data)
                        all_results[seed][item_mod][method_name] = {}
                        all_results[seed][item_mod][method_name]['Result'] = round(compute_exact_predictions(real_labels, result), 3)
        result_df = generate_table_with_std(all_results)
        result_df = result_df.round(decimals=4)
        os.makedirs('results', exist_ok=True)
        result_df.to_csv(f'results/real_data_results_multiple.csv')

