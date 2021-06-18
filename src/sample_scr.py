import pandas as pd
import numpy as np
import random
import os
import json
import fire
import shutil

def sample(path_to_data1, path_to_data2, in_words, all_words, n_pair=100, random_state=1, delete_short_and_long=False, delete_first_and_last_position=False, mode=None):
    words = os.listdir(path_to_data1)
    words2 = os.listdir(path_to_data2)
    i1 = words[0].split('_')[1].split('.')[0]
    i2 = words2[0].split('_')[1].split('.')[0]
    i12 = [i1, i2]
    #words = [word.split('_')[0] for word in words]
    words = [str(all_words.index(i)) for i in in_words]
    list_data = [path_to_data1, path_to_data2]
    for word in words:
        dfs = {}
        for i, temp in enumerate(list_data):
            dfs[temp] = pd.read_csv(temp + word + '_' + i12[i] + '.tsv', sep='\t', names=['sent', 'pos'], quoting=3, error_bad_lines=False)
            dfs[temp] = dfs[temp].drop_duplicates(subset=['sent'])
            dfs[temp]['pos'] = dfs[temp].apply(lambda r: eval(r.pos), axis=1)
            dfs[temp]['pos'] = dfs[temp].apply(lambda r: r.pos[0] if len(r.pos) == 1 else random.choice(r.pos), axis=1)
        if delete_short_and_long:
            smin = []
            smax = []
            for i in dfs:
                smin.append(np.quantile(np.array([len(i[1].sent) for i in dfs[i].iterrows()]), q=0.25))
                smax.append(np.quantile(np.array([len(i[1].sent) for i in dfs[i].iterrows()]), q=0.75))
            for j, i in enumerate(dfs):
                dfs[i]['sent'] = dfs[i].apply(lambda r: r.sent if len(r.sent) >= smin[j] else 0, axis=1)
                dfs[i] = dfs[i][dfs[i].sent != 0]
                dfs[i]['sent'] = dfs[i].apply(lambda r: r.sent if len(r.sent) <= smax[j] else 0, axis=1)
                dfs[i] = dfs[i][dfs[i].sent != 0]
        if delete_first_and_last_position:
            for j, i in enumerate(dfs):
                dfs[i]['sent'] = dfs[i].apply(lambda r: r.sent if r.pos[0] > 1 else 0, axis=1)
                dfs[i] = dfs[i][dfs[i].sent != 0]
                dfs[i]['sent'] = dfs[i].apply(lambda r: r.sent if r.pos[1] < len(r.sent) - 1 else 0, axis=1)
                dfs[i] = dfs[i][dfs[i].sent != 0]
        k_pair = []
        for i in dfs:
            if len(dfs[i]) < n_pair:
                k_pair.append(len(dfs[i]))
            else:
                k_pair.append(n_pair)

        sample_dfs = [dfs[j].sample(n=k_pair[i], random_state=random_state).reset_index().reset_index() for i, j in enumerate(dfs)]
        sample_dfs = [df.drop(columns=['index']) for df in sample_dfs]
        sample_dfs[0] = sample_dfs[0].rename(columns={'sent': 'sent1', 'pos': 'pos1'})
        sample_dfs[1] = sample_dfs[1].rename(columns={'sent': 'sent2', 'pos': 'pos2'})
        df12 = pd.concat([sample_dfs[0], sample_dfs[1]], axis=1)
        #df12_ = sample_dfs[0].merge(sample_dfs[1], on='level_0')
        df12 = df12.drop(columns=['level_0'])
        #df12_ = df12_.drop(columns=['level_0'])
        df12['word'] = all_words[int(word)]
        flag = mode
        if 'pairs' not in os.listdir():
            os.mkdir('pairs')
        if f'{flag}.scd_{i1}{i2}_sl-{delete_short_and_long}_fl-{delete_first_and_last_position}_np-{n_pair}' not in os.listdir('pairs'):
            os.mkdir(f'pairs/{flag}.scd_{i1}{i2}_sl-{delete_short_and_long}_fl-{delete_first_and_last_position}_np-{n_pair}')
        df12.to_csv(f'pairs/{flag}.scd_{i1}{i2}_sl-{delete_short_and_long}_fl-{delete_first_and_last_position}_np-{n_pair}/{i1}-{i2}_' + word + '.tsv', sep='\t', index=False)


def full_sample(paths_test, paths_dev, path_words_test, path_words_dev, in_words_dev1, in_words_dev2, delete_short_and_long, delete_first_and_last_position, n_pairs):
    in_words = list(pd.read_csv(path_words_test, sep='\t').word)
    sample(paths_test[0], paths_test[1], in_words, in_words, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='test')
    sample(paths_test[1], paths_test[2], in_words, in_words, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='test')
    sample(paths_test[0], paths_test[2], in_words, in_words, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='test')
    in_words_dev_1 = list(pd.read_csv(in_words_dev1, sep='\t').word)
    in_words_dev_2 = list(pd.read_csv(in_words_dev2, sep='\t').word)
    in_words_dev_1 = list(pd.read_csv('rushiftEval/raw_annotation_1_words', sep='\t', names=['word']).word)
    in_words_dev_2 = list(pd.read_csv('rushiftEval/raw_annotation_2_words', sep='\t', names=['word']).word)
    in_words = list(pd.read_csv(path_words_dev[0], sep='\t').word)
    sample(paths_dev[0], paths_dev[1], in_words, in_words_dev_1, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='train_1')
    in_words = list(pd.read_csv(path_words_dev[1], sep='\t').word)
    sample(paths_dev[2], paths_dev[3], in_words, in_words_dev_2, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='train_2')
    in_words = list(pd.read_csv(path_words_dev[2], sep='\t').word)
    sample(paths_dev[0], paths_dev[1], in_words, in_words_dev_1, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='dev_1')
    in_words = list(pd.read_csv(path_words_dev[3], sep='\t').word)
    sample(paths_dev[2], paths_dev[3], in_words, in_words_dev_2, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pair=n_pairs, mode='dev_2')

def combine_words(path_to_sampled):
    if 'combine' not in os.listdir(path_to_sampled):
        os.mkdir(path_to_sampled + 'combine')
    for i in sorted(os.listdir(path_to_sampled)):
        if i == 'combine':
            continue
        new_path = path_to_sampled + i +'/'
        df = pd.DataFrame()
        for j in sorted(os.listdir(new_path)):
            df_temp = pd.read_csv(new_path+j, sep='\t')
            df = pd.concat([df, df_temp], ignore_index=True)
        if i not in os.listdir(path_to_sampled+'combine'):
            os.mkdir(path_to_sampled+'combine/'+i)
        df.to_csv(path_to_sampled+'combine/'+i+'/'+'combine_words.tsv', sep='\t', index=False)

def tsv_to_json(path_to_combine):
    if 'dev_1.rusemshift' not in os.listdir(path_to_combine):
        os.mkdir(path_to_combine + 'dev_1.rusemshift')
        os.mkdir(path_to_combine + 'dev_2.rusemshift')
        os.mkdir(path_to_combine + 'train_1.rusemshift')
        os.mkdir(path_to_combine + 'train_2.rusemshift')
    shutil.copyfile('rushiftEval/dev_1_rusemshift', path_to_combine + 'dev_1.rusemshift/combine_words.tsv')
    shutil.copyfile('rushiftEval/dev_2_rusemshift', path_to_combine + 'dev_2.rusemshift/combine_words.tsv')
    shutil.copyfile('rushiftEval/train_1_rusemshift', path_to_combine + 'train_1.rusemshift/combine_words.tsv')
    shutil.copyfile('rushiftEval/train_2_rusemshift', path_to_combine + 'train_2.rusemshift/combine_words.tsv')
    for i in sorted(os.listdir(path_to_combine)):
        df = pd.read_csv(path_to_combine + i + '/combine_words.tsv', sep='\t')
        df['pos1'] = df.apply(lambda r: eval(r.pos1) if type(r.pos1) == type('str') else 0, axis=1)
        df['pos2'] = df.apply(lambda r: eval(r.pos2) if type(r.pos2) == type('str') else 0, axis=1)
        df = df[df.pos1 != 0]
        df = df[df.pos2 != 0]
        json_file = [
            {'id': f'{i}.scd.{row[0]}', 'lemma': row[1].word, 'pos': 'NOUN', 'sentence1': row[1].sent1,
             'sentence2': row[1].sent2, 'start1': row[1].pos1[0], 'end1': row[1].pos1[1], 'start2': row[1].pos2[0],
             'end2': row[1].pos2[1], 'grp': 'COMPARE'} for row in df.iterrows()]
        file_out = path_to_combine + i + '/combine_words.data'
        f = open(file_out, 'w')
        json.dump(json_file, f, indent=4)

def to_one_dir(path):
    if 'sampled_data' not in os.listdir(path):
        os.mkdir(path + 'sampled_data')
    for i in os.listdir(path):
        if i == 'sampled_data':
            continue
        new_path = path + i + '/'
        shutil.copyfile(new_path + 'combine_words.data', path + 'sampled_data/' + i + '.data')


def all_sample(path_to_test, path_to_dev, path_to_sampled, path_to_combine, in_words_dev1, in_words_dev2, delete_short_and_long, delete_first_and_last_position, n_pairs):
    full_sample(paths_test=[path_to_test+'eval_1/', path_to_test+'eval_2/', path_to_test+'eval_3/'],
                    paths_dev=[path_to_dev+'1_1_tsv/', path_to_dev+'1_2_tsv/', path_to_dev+'2_2_tsv/', path_to_dev+'2_3_tsv/'],
                    path_words_test='rushiftEval/new_eval_answer.tsv',
                    path_words_dev=['rushiftEval/words_train_1.tsv', 'rushiftEval/words_train_2.tsv', 'rushiftEval/words_dev_1.tsv', 'rushiftEval/words_dev_2.tsv'],
                    in_words_dev1=in_words_dev1, in_words_dev2=in_words_dev2, delete_short_and_long=delete_short_and_long, delete_first_and_last_position=delete_first_and_last_position, n_pairs=n_pairs)
    combine_words(path_to_sampled)
    tsv_to_json(path_to_combine)
    to_one_dir(path_to_combine)


if __name__ == '__main__':
    fire.Fire(all_sample)
