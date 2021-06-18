import pandas as pd
import numpy as np
from typing import List
from scipy.stats import spearmanr
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.datasets import load_data

def gen_all_combin(methods: List[str]):
    def gen_all_combin1(methods: List[str]):
        if len(methods) == 1:
            return [[], methods]
        else:
            gen1 = gen_all_combin1(methods[1:])
            return gen1 + [methods[:1]+comb for comb in gen1]
    assert len(methods) > 0, 'List must be not empty!'
    if len(methods) > 25:
        print('WARNING! gen_all_combin(): risk of getting out of memory')
    return gen_all_combin1(methods)[1:]

def aggregate_by_word(df, score_agg_func):
    mask = (df.annotator1 != 0) & (df.annotator2 != 0) & (df.annotator3 != 0) & \
        (df.annotator4 != 0) & (df.annotator5 != 0)
    df_mean = df[mask][['word','mean']].groupby('word').agg('mean').reset_index()
    df_score = df[mask][['word','score']].groupby('word').agg(score_agg_func).reset_index()
    return df_mean.merge(df_score)

def comb_one_score(score1, score2, method_comb):
    if method_comb == '*':
        return score1 * score2
    elif method_comb == '+':
        return score1 + score2
    
def make_comb_scores(dfs, methods, method_comb=None):
    # -> pd.DataFrame({1: df, 2: df})
    df_comb_scores = {}
    for n in (1,2):
        df_comb_scores[n] = dfs[f'{methods[0]}_{n}'].copy()
        if len(methods) > 1:
            df_comb_scores[n]['score1'] = df_comb_scores[n].score
            df_comb_scores[n] = df_comb_scores[n].drop('score',axis=1)
            for method in methods[1:]:
                df_comb_scores[n] = df_comb_scores[n].merge(dfs[f'{method}_{n}'])
                df_comb_scores[n].score1 = df_comb_scores[n].apply(lambda r: comb_one_score(r.score1, r.score, method_comb), axis=1)
                df_comb_scores[n] = df_comb_scores[n].drop('score', axis=1)
            df_comb_scores[n] = df_comb_scores[n].rename(columns={'score1': 'score'})
    return df_comb_scores

def split_by_train_dev(comb_scores, data, combin, enable_warning=True):
    new_data = {}
    for split in ('train', 'dev'):
        for n in (1,2):
            cnt_examples = len(data[f'{split}_{n}'])
            new_data[f'{split}_{n}'] = data[f'{split}_{n}'].merge(comb_scores[n]).dropna()
            cnt_nan = cnt_examples - len(new_data[f'{split}_{n}'])
            if cnt_nan > 0 and enable_warning:
                print(f'WARNING! From {combin}, {split}_{n} dropped {cnt_nan} NaN scores ({cnt_nan/cnt_examples*100:.2f}%)')
    return new_data


def get_metrics(df_scores, score_agg_func, max_neg_score=2., min_true_score=3.):
    """
    acc_wic - точность для пар предложений, у которых (mean < max_neg_score or mean > min_true_score)
              Считается 10-фолдовая кросс-валидиация для линейной регрессии с одним признаком (score) 
              и среди них берется минимальная точность
    """
    df_cols = {'metric': ['corr_word','corr_wic','acc_wic'], 'dev_1':[], 'dev_2':[], 'train_1':[], 'train_2':[]}
    for name, df in df_scores.items():
        corr_wic, _ = spearmanr(df.score, df['mean'])
        
        df_acc = df[(df['mean'] < max_neg_score) | (df['mean'] > min_true_score)]
        if not df_acc.empty:
            acc_wic = min(cross_val_score(LogisticRegression(C=10), df_acc[['score']], df_acc['mean'] > min_true_score, 
                  cv=10))
        else:
            acc_wic = np.nan

        df_word = aggregate_by_word(df, score_agg_func)
        corr_word, _ = spearmanr(df_word.score, df_word['mean'])
        df_cols[name] += [corr_word, corr_wic, acc_wic]
    return pd.DataFrame(df_cols)

def evaluate_scores(paths_to_scores, path_save=None, enable_warning=True, only_compare=True, return_df=True):
    # paths_to_scores = {'a0': 'Adis_rusemshift', 'm': 'maks_base'}
    if path_save is None:
        path_save = '-'.join(paths_to_scores.keys()) + '.evaluate_scores.tsv'
    scores_dir = 'scores/'
    
    dfs = {f'{name}_{n}': pd.read_csv(scores_dir + paths_to_scores[name] + f'_{n}_scores.tsv', sep='\t', converters={'pos1':eval, 'pos2': eval}) 
           for name in paths_to_scores for n in (1,2)}
    data = load_data('rusemshift')
    data.pop('train_comb_rusemshift')
    data = {name[:-len('_rusemshift')]: df for name, df in data.items()}
    if only_compare:
        for name in data:
            if 'group' in data[name]:
                data[name] = data[name][data[name].group == 'COMPARE']
    """ Разбиваем на несколько примеров примеры, где в pos1/pos2 список из нескольких туплов """
    for name in data:
        for n in (1,2):
            s = data[name][f'pos{n}'].apply(pd.Series, 1).stack()
            s.index = s.index.droplevel(-1) # to line up with df's index
            s.name = f'pos{n}'
            data[name] = data[name].drop(f'pos{n}',axis=1).join(s)
        data[name] = data[name].drop_duplicates()
    
    combinations = gen_all_combin(list(paths_to_scores.keys()))
    df_metric = pd.DataFrame()
    for combin in combinations:
        methods_comb = (None,) if len(combin) == 1 else ('*', '+')
        for method_comb in methods_comb:
            comb_scores = make_comb_scores(dfs, combin, method_comb=method_comb)
            new_data = split_by_train_dev(comb_scores, data, combin, enable_warning)
            agg_funcs = {f'quant_{quant*0.1}': (lambda x: np.quantile(x, quant*0.1)) for quant in range(11)}
            agg_funcs.update({'mean': (lambda x: x.mean())})
            for name_agg_func, agg_func in agg_funcs.items():
                df_one = get_metrics(new_data, agg_func)
                df_one['agg_func'] = name_agg_func
                df_one['method_comb'] = method_comb
                df_one['comb_scores'] = [combin] * len(df_one)
                df_metric = df_metric.append(df_one, ignore_index=True)
    df_metric.to_csv(path_save, sep='\t', index=False)
    if return_df:
        return df_metric
    else:
        return

from fire import Fire
if __name__ == '__main__':
	Fire(evaluate_scores)

