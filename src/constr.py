import pandas as pd
import numpy as np
import fire
import json
import os

def func(r, id, scores):
    if r.id not in id:
        return -1
    return scores[id.index(r.id)]


def constr(path_to_data='pairs/combine/sampled_data/',
           path_to_scores='sampled_data/score/',
           path_to_ans='sampled_data/ans/',
           path_to_gold='gold/',
           model_name='',
           type='mean'):
    """
    constructs the required (convenient) file format for calculating metrics, creates tsv files that need
    to the function for calculating metrics
    :param path_to_data: path to the directory with json files `.data` where pairs of sentences are
        located for all samples
    :param path_to_scores: path to the directory with json files `.scores` where scores for sentences
        predicted by wic model are located for all samples
    :param path_to_ans: path to the directory with json files where tags('T' or 'F') for sentences
        predicted by wic model are located for all samples
    :param path_to_gold: path to directory with json files where gold tags('T' or 'F') and gold scores for
        sentences for all samples
    :param model_name: name of wic model for correct save
    :param type: wic model return list of 2 scores. this parameter has 4 options:
        for predicted [a,b] by wic model
        1) 'mean' - (a + b) /2
        2) '1' - a
        3) '2' - b
        4) 'geom' - sqrt(a*b)
    :return: void
    """
    if model_name+'_tsvs' not in os.listdir(path_to_data):
        os.mkdir(path_to_data+model_name+'_tsvs')
    if model_name+'_tsvs' not in os.listdir(path_to_gold):
        os.mkdir(path_to_gold+model_name+'_tsvs')

    for i in os.listdir(path_to_data):
        if i.endswith('tsvs'):
            continue
        f = open(path_to_data + i, 'r')
        f2 = open(path_to_scores + i[:-4]+'scd.scores', 'r')
        f3 = open(path_to_ans + i[:-4]+'scd', 'r')
        data = json.load(f)
        data2 = json.load(f2)
        data3 = json.load(f3)
        temp_df1 = pd.DataFrame()
        temp_df2 = pd.DataFrame()
        temp_df3 = pd.DataFrame()
        words, id, sent1, sent2, pos1, pos2, grp = [], [], [], [], [], [], []
        id_scores, scores = [], []
        id_ans, anss = [], []
        for j in data:
            words.append(j['lemma'])
            id.append(j['id'])
            sent1.append(j['sentence1'])
            sent2.append(j['sentence2'])
            pos1.append((j['start1'], j['end1']))
            pos2.append((j['start2'], j['end2']))
            grp.append(j['grp'])
        for j in data2:
            s = j['score']
            s = np.array([float(q) for q in s])
            if len(s) != 2:
                continue
            id_scores.append(j['id'])
            if type == 'mean':
                scores.append(s.mean())
            elif type == '1':
                scores.append(s[0])
            elif type == '2':
                scores.append(s[1])
            elif type == 'geom':
                scores.append(np.sqrt(s[0] * s[1]))
        for j in data3:
            id_ans.append(j['id'])
            anss.append(j['tag'])
        temp_df1['word'] = words
        temp_df1['sent1'] = sent1
        temp_df1['sent2'] = sent2
        temp_df1['id'] = id
        temp_df1['pos1'] = pos1
        temp_df1['pos2'] = pos2
        temp_df1['grp'] = grp
        temp_df2['id'] = id_scores
        temp_df2['scores'] = scores
        temp_df3['id'] = id_ans
        temp_df3['tag'] = anss
        temp_df = temp_df1.merge(temp_df2, on='id', how='inner')
        temp_df = temp_df.merge(temp_df3, on='id', how='inner')
        temp_df.to_csv(path_to_data + model_name + '_tsvs/' + i[:-4] + 'tsv', sep='\t', index=False)
    for i in os.listdir(path_to_gold):
        if i.endswith('tsvs'):
            continue
        f = open(path_to_gold + i, 'r')
        data_scores = json.load(f)
        temp_df = pd.DataFrame()
        scores, anses, id = [], [], []
        for j1 in data_scores:
            s = j1['score']
            scores.append(s)
            id.append(j1['id'])
            anses.append(j1['tag'])
        temp_df['id'] = id
        temp_df['tag'] = anses
        temp_df['score'] = scores
        temp_df.to_csv(path_to_gold + model_name + '_tsvs/' + i + '.tsv', sep='\t', index=False)

if __name__ == '__main__':
    fire.Fire(constr)
