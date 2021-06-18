import pandas as pd
import fire
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

path_to_data = 'pairs/combine/sampled_data/tsvs/'
path_to_gold = 'gold/tsvs/'

def spearman(y_true_score, y_pred_score):
	corr, _ = spearmanr(y_true_score, y_pred_score)
	return corr

def FitIsotonicRegression(X_train, y_train, X_test, out_of_bounds='clip', verbose=False):
    iso_reg = IsotonicRegression(y_min=1., y_max=4., increasing='auto', out_of_bounds=out_of_bounds).fit(X_train, y_train)
    if verbose:
        y_pred = iso_reg.predict(X_train)
        corr, _ = spearman(y_pred, y_train)
        corr_wo_train, _ = spearman(X_train, y_train)
        print(f'Corr on train = {corr:.4f}, w/o isotonic regr = {corr_wo_train:.4f}')
    return iso_reg.predict(X_test)


def fmean(df_temp):
    qq = df_temp.groupby('word').mean()
    qq.sort_index(inplace=True)
    return list(qq.index), list(qq.scores)

def acc1(x, y):
    return accuracy_score(x, y)

def run_model(train_gold=[(path_to_data + 'train_1.rusemshift.tsv', path_to_gold + 'train_1.rusemshift.gold.tsv'),
                          (path_to_data + 'train_2.rusemshift.tsv', path_to_gold + 'train_2.rusemshift.gold.tsv')],
              dev=[path_to_data + 'dev_1.rusemshift.tsv', path_to_data + 'dev_2.rusemshift.tsv',
                   path_to_data + 'dev_1.scd_12_sl-True_fl-True_np-100.tsv'],
              dev_gold=[(path_to_data + 'dev_1.rusemshift.tsv', path_to_gold + 'dev_1.rusemshift.gold.tsv'),
                        (path_to_data + 'dev_2.rusemshift.tsv', path_to_gold + 'dev_2.rusemshift.gold.tsv')],
              test=[path_to_data + 'test.scd_12_sl-True_fl-True_np-100.tsv',
                    path_to_data + 'test.scd_23_sl-True_fl-True_np-100.tsv',
                    path_to_data + 'test.scd_13_sl-True_fl-True_np-100.tsv'],	
              model_name=''):
    """
    launches a model that calculates metrics using isotonic regression on the scores for sentences and saves the metrics
    on the dev sample and prepares a file for submission. Saves 2 files: with dev metrics and a file for submission
    :param train_gold: list of tuples (path_to_train_1, path_to_train_2_gold) (path_to_train_1, path_to_train_2_gold)
    :param dev: list of paths to tsv files with dev_data
    :param dev_gold: list of tuples (path_to_dev_1, path_to_dev_2_gold) (path_to_dev_1, path_to_dev_2_gold)
    :param test: list of paths to tsv files with test_data
    :param model_name: wic-model name
    :return:
    """
    X_train, y_train = [], []
    for j in train_gold:
        df_temp_ans = pd.read_csv(j[1], sep='\t')
        df_temp_ans_data = pd.read_csv(j[0], sep='\t')
        df_temp_ans = df_temp_ans.rename(columns={'score': 'gold_score', 'tag': 'gold_tag'})
        df_temp_ans_data = df_temp_ans_data.merge(df_temp_ans, on='id', how='inner')
        X_train.extend(list(df_temp_ans_data.scores))
        y_train.extend(list(df_temp_ans_data.gold_score))
    df_dev = pd.DataFrame()
    t = [a[0].split('/')[-1] for a in train_gold]
    t = "+".join(t)
    df_dev['model'] = [f'iso_{model_name}_trained-{t}']
    devs_gold, dfs_dev_ans = {}, {}
    for i in dev_gold:
        df_temp_ans = pd.read_csv(i[1], sep='\t')
        df_temp_ans_data = pd.read_csv(i[0], sep='\t')
        df_temp_ans = df_temp_ans.rename(columns={'score': 'gold_score', 'tag': 'gold_tag'})
        df_temp_ans_data = df_temp_ans_data.merge(df_temp_ans, on='id', how='inner')
        if '_1.' in i[0]:
            temp = 'dev_1'
        else:
            temp = 'dev_2'
        dfs_dev_ans[temp] = df_temp_ans_data
    for i in dev:
        df_temp = pd.read_csv(i, sep='\t')
        X_dev = df_temp.scores
        pred = FitIsotonicRegression(X_train, y_train, X_dev)
        df_temp['iso_score'] = pred
        qq = df_temp.groupby('word').mean()
        if '_1.' in i:
            y_cor = list(dfs_dev_ans['dev_1'].groupby('word').mean().gold_score)
        else:
            y_cor = list(dfs_dev_ans['dev_2'].groupby('word').mean().gold_score)
        corr = spearman(y_cor, list(qq.iso_score))
        if 'fl' not in i and 'sl' not in i and 'np' not in i:
            if '_1.' in i:
                df_temp_ans_data = dfs_dev_ans['dev_1']
            else:
                df_temp_ans_data = dfs_dev_ans['dev_2']
            sent_spearman = spearman(df_temp_ans_data.gold_score, X_dev)
            acc = acc1(list(df_temp_ans_data.tag), list(df_temp_ans_data.gold_tag))
            df_dev[i.split('/')[-1] + '_accuracy'] = [acc]
            df_dev[i.split('/')[-1] + '_sent_spearman'] = [sent_spearman]
        df_dev[i.split('/')[-1] + '_word_spearman'] = [corr]
    df_dev.to_csv(f'{model_name}_trained-{t}_iso_dev_metrics.tsv', sep='\t', index=False, header=True)
    df_subm = pd.DataFrame()
    for i in test:
        df_temp = pd.read_csv(i, sep='\t')
        X_test = df_temp.scores
        pred = FitIsotonicRegression(X_train, y_train, X_test)
        df_temp['iso_score'] = pred
        qq = df_temp.groupby('word').mean()
        words = list(qq.index)
        scores = list(qq.iso_score)
        if len(df_subm) == 0:
            df_subm['word'] = words
        df_subm[i.split('/')[-1]] = scores
    temp = df_subm.columns[1]
    df_subm.to_csv(f'{model_name}_trained-{t}_iso_answer_{temp.split(".")[1]}.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    fire.Fire(run_model)
