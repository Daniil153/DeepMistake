import pandas as pd
import fire
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
path_to_data = 'pairs/combine/sampled_data/tsvs/'
path_to_gold = 'gold/tsvs/'

def spearman(y_true_score, y_pred_score):
	corr, _ = spearmanr(y_true_score, y_pred_score)
	return corr

def fstat(df_temp, stat):
    fch = {}
    for st in stat:
        if st == 'mean':
            qq = df_temp.groupby('word').mean()
        else:
            qq = df_temp.groupby('word').quantile(q=float(st))
        qq.sort_index(inplace=True)
        if len(fch) == 0:
            for q in qq.iterrows():
                fch[q[0]] = [q[1].scores]
        for q in qq.iterrows():
            fch[q[0]].append(q[1].scores)
    return fch

def acc1(x, y):
    return accuracy_score(x, y)

def run_model(train=[path_to_data + 'train_1.rusemshift.tsv', path_to_data + 'train_2.rusemshift.tsv'],
              train_gold=[(path_to_data + 'train_1.rusemshift.tsv', path_to_gold + 'train_1.rusemshift.gold.tsv'),
                          (path_to_data + 'train_2.rusemshift.tsv', path_to_gold + 'train_2.rusemshift.gold.tsv')],
              dev=[path_to_data + 'dev_1.rusemshift.tsv', path_to_data + 'dev_2.rusemshift.tsv',
                   path_to_data + 'dev_1.scd_12_sl-True_fl-True_np-100.tsv'],
              dev_gold=[(path_to_data + 'dev_1.rusemshift.tsv', path_to_gold + 'dev_1.rusemshift.gold.tsv'),
                        (path_to_data + 'dev_2.rusemshift.tsv', path_to_gold + 'dev_2.rusemshift.gold.tsv')],
              test=[path_to_data + 'test.scd_12_sl-True_fl-True_np-100.tsv',
                    path_to_data + 'test.scd_23_sl-True_fl-True_np-100.tsv',
                    path_to_data + 'test.scd_13_sl-True_fl-True_np-100.tsv'],
              stat=['mean', '0.25', '0.5', '0.75'],
              model_name=''):
    """
    launches a model that calculates metrics using linear regression on the statistics and saves the metrics
    on the dev sample and prepares a file for submission. Saves 2 files: with dev metrics and a file for submission
    :param train: list of paths to tsv files with train_data
    :param train_gold: list of tuples (path_to_train_1, path_to_train_2_gold) (path_to_train_1, path_to_train_2_gold)
    :param dev: list of paths to tsv files with dev_data
    :param dev_gold: list of tuples (path_to_dev_1, path_to_dev_2_gold) (path_to_dev_1, path_to_dev_2_gold)
    :param test: list of paths to tsv files with test_data
    :param stat: list of `mean` and statistics in str format (0.25)
    :param model_name: wic-model name
    :return: void
    """
    lr = LinearRegression()
    parameters = {}
    reg = GridSearchCV(lr, parameters, cv=10)
    X_train, y_train = [], []
    trains_gold, dfs_train_ans = {}, {}
    for i in train_gold:
        df_temp_ans = pd.read_csv(i[1], sep='\t')
        df_temp_ans_data = pd.read_csv(i[0], sep='\t')
        df_temp_ans = df_temp_ans.rename(columns={'score': 'gold_score', 'tag': 'gold_tag'})
        df_temp_ans_data = df_temp_ans_data.merge(df_temp_ans, on='id', how='inner')
        qq_ans = df_temp_ans_data.groupby('word').mean()
        qq_ans.sort_index(inplace=True)
        df_temp_ans_data['tag'] = df_temp_ans_data.apply(lambda r: 'T' if r.scores > 0.5 else 'F', axis=1)
        if '_1.' in i[0]:
            temp = 'train_1'
        else:
            temp = 'train_2'
        trains_gold[temp] = qq_ans
        dfs_train_ans[temp] = df_temp_ans_data
    for i in train:
        df_temp = pd.read_csv(i, sep='\t')
        fch = fstat(df_temp, stat)
        X_train.extend(list(fch.values()))
        if '_1.' in i:
            y_train_temp = list(trains_gold['train_1'].gold_score)
        else:
            y_train_temp = list(trains_gold['train_2'].gold_score)
        y_train.extend(y_train_temp)
    reg.fit(X_train, y_train)
    df_dev = pd.DataFrame()
    t = [a.split('/')[-1] for a in train]
    df_dev['model'] = [f'lr_{stat}_{model_name}_trained-{"+".join(t)}']
    devs_gold, dfs_dev_ans = {}, {}
    for i in dev_gold:
        df_temp_ans = pd.read_csv(i[1], sep='\t')
        df_temp_ans_data = pd.read_csv(i[0], sep='\t')
        df_temp_ans = df_temp_ans.rename(columns={'score': 'gold_score', 'tag': 'gold_tag'})
        df_temp_ans_data = df_temp_ans_data.merge(df_temp_ans, on='id', how='inner')
        qq_ans = df_temp_ans_data.groupby('word').mean()
        qq_ans.sort_index(inplace=True)
        df_temp_ans_data['tag'] = df_temp_ans_data.apply(lambda r: 'T' if r.scores > 0.5 else 'F', axis=1)
        if '_1.' in i[0]:
            temp = 'dev_1'
        else:
            temp = 'dev_2'
        devs_gold[temp] = qq_ans
        dfs_dev_ans[temp] = df_temp_ans_data
    for i in dev:
        df_temp = pd.read_csv(i, sep='\t')
        fch = fstat(df_temp, stat)
        X_dev = list(fch.values())
        if '_1.' in i:
            y_dev = list(devs_gold['dev_1'].gold_score)
        else:
            y_dev = list(devs_gold['dev_2'].gold_score)
        pred = reg.predict(X_dev)
        corr = spearman(y_dev, pred)
        if 'fl' not in i and 'sl' not in i and 'np' not in i:
            if '_1.' in i:
                df_temp_ans_data = dfs_dev_ans['dev_1']
            else:
                df_temp_ans_data = dfs_dev_ans['dev_2']
            sent_spearman = spearman(df_temp_ans_data.gold_score, df_temp_ans_data.scores)
            acc = acc1(list(df_temp_ans_data.tag), list(df_temp_ans_data.gold_tag))
            df_dev[i.split('/')[-1] + '_accuracy'] = [acc]
            df_dev[i.split('/')[-1] + '_sent_spearman'] = [sent_spearman]
        df_dev[i.split('/')[-1] + '_word_spearman'] = [corr]
    df_dev.to_csv(f'{model_name}_trained-{"+".join(t)}_stat_dev_metrics.tsv', sep='\t', index=False, header=True)
    df_subm = pd.DataFrame()
    for i in test:
        df_temp = pd.read_csv(i, sep='\t')
        fch = fstat(df_temp, stat)
        X_test = list(fch.values())
        words = list(fch.keys())
        pred = reg.predict(X_test)
        if len(df_subm) == 0:
            df_subm['word'] = words
        df_subm[i.split('/')[-1]] = pred
    temp = df_subm.columns[1]
    df_subm.to_csv(f'{model_name}_trained-{"+".join(t)}_stat_answer_{temp.split(".")[1]}.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    fire.Fire(run_model)
