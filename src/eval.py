import pandas as pd
import json
import os
from scipy.stats import spearmanr
import fire


def spearman(y_true_score, y_pred_score):
    corr, _ = spearmanr(y_true_score, y_pred_score)
    return corr


def check_test(f1, f2, name=['']):
    """
    receives 2 paths: to the file with the predicted scores of the model and to the file with the gold scores, save 3
    results of the Spearman correlation by words
    :param f1: path to the file with the predicted scores
    :param f2: path to the file with the gold scores
    :param name: list of name model, type of sampled data, method and etc. for correct save results
    :return: void
    """
    temp = '_'.join(name)
    df_pred = pd.read_csv(f1, sep='\t', names=['word', '12_pred', '23_pred', '13_pred'])
    df_gold = pd.read_csv(f2, sep='\t', names=['word', '12_gold', '23_gold', '13_gold'])
    df = df_pred.merge(df_gold, on='word', how='inner')
    ans12 = spearman(df['12_pred'], df['12_gold'])
    ans23 = spearman(df['23_pred'], df['23_gold'])
    ans13 = spearman(df['13_pred'], df['13_gold'])
    df_ans = pd.DataFrame()
    df_ans['name'] = [temp]
    df_ans['12_res'] = [ans12]
    df_ans['23_res'] = [ans23]
    df_ans['13_res'] = [ans13]
    avg = (ans12 + ans23 + ans13) / 3
    df_ans['avg_res'] = [avg]
    df_ans.to_csv(f'results_{temp}', sep='\t', index=False)


if __name__ == '__main__':
    fire.Fire(check_test)
