import pandas as pd
import fire
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

path_to_data = 'pairs/combine/sampled_data/'
path_to_gold = 'gold/'

def spearman(y_true_score, y_pred_score):
	corr, _ = spearmanr(y_true_score, y_pred_score)
	return corr

def fmean(df_temp):
    qq = df_temp.groupby('word').mean()
    qq.sort_index(inplace=True)
    return list(qq.index), list(qq.scores)

def acc1(x, y):
    return accuracy_score(x, y)

def run_model(dev=['dev_1.rusemshift.tsv', 'dev_2.rusemshift.tsv',
                   'dev_1.scd_12_sl-True_fl-True_np-100.tsv'],
              dev_gold=[('dev_1.rusemshift.tsv', 'dev_1.rusemshift.gold.tsv'),
                        ('dev_2.rusemshift.tsv', 'dev_2.rusemshift.gold.tsv')],
              test=['test.scd_12_sl-True_fl-True_np-100.tsv',
                    'test.scd_23_sl-True_fl-True_np-100.tsv',
                    'test.scd_13_sl-True_fl-True_np-100.tsv'],
              test_gold='rushiftEval/eval_answer.tsv',
              model_name=''):
    dev = [path_to_data + model_name + '_tsvs/' + i for i in dev]
    dev_gold = [(path_to_data + model_name + '_tsvs/' + i, path_to_gold + model_name + '_tsvs/' + j) for i,j in dev_gold]
    test = [path_to_data + model_name + '_tsvs/' + i for i in test]
    run_method(dev=dev,
              dev_gold=dev_gold,
              test=test,
              test_gold=test_gold,
              model_name=model_name)


def run_method(dev=[path_to_data + 'dev_1.rusemshift.tsv', path_to_data + 'dev_2.rusemshift.tsv',
                   path_to_data + 'dev_1.scd_12_sl-True_fl-True_np-100.tsv'],
              dev_gold=[(path_to_data + 'dev_1.rusemshift.tsv', path_to_gold + 'dev_1.rusemshift.gold.tsv'),
                        (path_to_data + 'dev_2.rusemshift.tsv', path_to_gold + 'dev_2.rusemshift.gold.tsv')],
              test=[path_to_data + 'test.scd_12_sl-True_fl-True_np-100.tsv',
                    path_to_data + 'test.scd_23_sl-True_fl-True_np-100.tsv',
                    path_to_data + 'test.scd_13_sl-True_fl-True_np-100.tsv'],
              test_gold='rushiftEval/eval_answer.tsv',
              model_name=''):
    """
    launches a model that calculates metrics using mean on the scores for sentences and saves the metrics
    on the dev sample and prepares a file for submission. Saves 2 files: with dev metrics and a file for submission
    :param dev: list of paths to tsv files with dev_data
    :param dev_gold: list of tuples (path_to_dev_1, path_to_dev_2_gold) (path_to_dev_1, path_to_dev_2_gold)
    :param test: list of paths to tsv files with test_data
    :param model_name: wic-model name
    :return: void
    """
    df_dev = pd.DataFrame()
    df_dev['model'] = [f'mean_{model_name}']
    devs_gold = {}
    dfs_ans = {}
    for i in dev_gold:
        df_temp_ans = pd.read_csv(i[1], sep='\t')
        df_temp_ans_data = pd.read_csv(i[0], sep='\t')
        df_temp_ans = df_temp_ans.rename(columns={'score': 'gold_score', 'tag': 'gold_tag'})
        df_temp_ans_data = df_temp_ans_data.merge(df_temp_ans, on='id', how='inner')
        qq_ans = df_temp_ans_data.groupby('word').mean()
        qq_ans.sort_index(inplace=True)
        df_temp_ans_data['tag'] = df_temp_ans_data.apply(lambda r: 'T' if r.scores > 0.5 else 'F', axis=1)
        if '_1.' in i[0].split('/')[-1]: 
            temp = 'dev_1'
        else:
            temp = 'dev_2'
        devs_gold[temp] = qq_ans
        dfs_ans[temp] = df_temp_ans_data
    for i in dev:
        df_temp = pd.read_csv(i, sep='\t')
        words, fch = fmean(df_temp)
        X_dev = fch
        if '_1.' in i:
            y_dev = list(devs_gold['dev_1'].gold_score)
        else:
            y_dev = list(devs_gold['dev_2'].gold_score)
        corr = spearman(y_dev, X_dev)
        if 'fl' not in i and 'sl' not in i and 'np' not in i:
            if '_1.' in i:
                df_temp_ans_data = dfs_ans['dev_1']
            else:
                df_temp_ans_data = dfs_ans['dev_2']
            sent_spearman = spearman(df_temp_ans_data.gold_score, df_temp_ans_data.scores)
            acc = acc1(list(df_temp_ans_data.tag), list(df_temp_ans_data.gold_tag))
            df_dev[i.split('/')[-1] + '_accuracy'] = [acc]
            df_dev[i.split('/')[-1] + '_sent_spearman'] = [sent_spearman]
        df_dev[i.split('/')[-1] + '_word_spearman'] = [corr]
    ppath = model_name.replace('+', '/')
    df_subm = pd.DataFrame()
    dfs_test_ans = {}
    for i in test:
        df_temp = pd.read_csv(i, sep='\t')
        fch = fmean(df_temp)
        words, X_test = fch
        if len(df_subm) == 0:
            df_subm['word'] = words
        df_subm[i.split('/')[-1]] = X_test
    if test_gold:
        columns_subm = list(df_subm.columns)
        df_subm = df_subm.rename(columns={columns_subm[1]: '12_pred', columns_subm[2]: '23_pred', columns_subm[3]: '13_pred'})
        df_an = pd.read_csv(test_gold, sep='\t', names=['word', '12_gold', '23_gold', '13_gold'])
        df_an = df_an.merge(df_subm, on='word', how='inner')
        ans12 = spearman(df_an['12_pred'], df_an['12_gold'])
        ans23 = spearman(df_an['23_pred'], df_an['23_gold'])
        ans13 = spearman(df_an['13_pred'], df_an['13_gold'])
        df_dev[f'sampled_test12_w_spearman'] = [ans12]
        df_dev[f'sampled_test13_w_spearman'] = [ans13]
        df_dev[f'sampled_test23_w_spearman'] = [ans23]
        df_dev[f'avg_test_w_spearman'] = [(ans13 + ans12 + ans23) / 3]
	print(f'Mean method: test12 - {ans12}, test13 - {ans13}, test23 - {ans23}, avg_test - {(ans13 + ans12 + ans23) / 3}')
    temp = df_subm.columns[1]
    df_dev.to_csv(f'{model_name}_mean_dev_metrics.tsv', sep='\t', index=False, header=True)
    df_subm.to_csv(f'{model_name}_mean_answer.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    fire.Fire(run_model)
