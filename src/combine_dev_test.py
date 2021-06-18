import pandas as pd
import numpy as np

def combine_dev_test(test_df, dev1_df, dev2_df):
    pass

def sample(words, n_pair=100, random_state=1):
    for word in words:
        dfs = [pd.read_csv(str(word) + str(i) + '.tsv') for i in [1, 2, 3]]
        sample_dfs = [df.sample(n=n_pair, random_state=random_state) for df in dfs]
        sample_dfs[0] = sample_dfs[0].rename(columns={'sent': 'sent1', 'pos': 'pos1'})
        sample_dfs[1] = sample_dfs[1].rename(columns={'sent': 'sent2', 'pos': 'pos2'})
        sample_dfs[2] = sample_dfs[2].rename(columns={'sent': 'sent2', 'pos': 'pos2'})
        df12 = pd.concat([sample_dfs[0], sample_dfs[1]], axis=1)
        df13 = pd.concat([sample_dfs[0], sample_dfs[2]], axis=1)
        sample_dfs[1] = sample_dfs[1].rename(columns={'sent2': 'sent1', 'pos2': 'pos1'})
        df23 = pd.concat([sample_dfs[1], sample_dfs[2]], axis=1)
        df12['word'] = word
        df13['word'] = word
        df23['word'] = word
        df12.to_csv('1-2_' + str(word) + '.tsv', sep='\t')
        df13.to_csv('1-3_' + str(word) + '.tsv', sep='\t')
        df23.to_csv('2-3_' + str(word) + '.tsv', sep='\t')


