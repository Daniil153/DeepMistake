import os
import pandas as pd
from pathlib import Path
import re

def extract_spans(string):
        regex = re.compile(r'(?P<start><b><i>)(?P<word>\w+)(?P<nonalpha>\W*)(?P<end></i></b>)')
        spans = []
        while True:
            match = regex.search(string)
            if match is None:
                break
            start = len(string[:match.start('start')])
            end = match.end('word') - len('<b><i>')
            string = \
                string[:match.start('start')] + \
                string[match.start('word'):match.end('nonalpha')] + \
                string[match.end('end'):]
            spans.append((start, end))
        return spans

def load_data(names_datasets=None, drop_blank_position=True):
    """
    :param names_datasets: dataset name or list of datasets names, e.g. 'mcl-wic', by default = list of all datasets
    :return: dict{split_name_dataset: pd.DataFrame}, e.g. {'train_comb_rusemshift': df1, 'dev_mcl-wic_en-ru': df2, ...}
                     format split_name_dataset: <split>_<params_dataset>_<name_dataset>
    """
    if names_datasets is None:
        names_datasets = ['mcl-wic', 'rusemshift']
    elif type(names_datasets) == str:
        names_datasets = [names_datasets]
    datasets_dir = str(Path(__file__).parent.parent.absolute()) + '/datasets'
    data = {}
    for name_data in names_datasets:
        if name_data == 'mcl-wic':
            full_splits = {'dev', 'test', 'train', 'training'}    # 'training' in mcl-wic; 'train' in rusemshift
            full_splits_mcl_wic = {'crosslingual', 'multilingual'}
            splits = set(os.listdir(f'{datasets_dir}/{name_data}')) & full_splits
            for split in splits:
                path_mcl_wic = f'{datasets_dir}/{name_data}/{split}'
                if split == 'training':
                    langs = 'en-en'
                    df = pd.read_json(f'{path_mcl_wic}/{split}.{langs}.data', orient='records')
                    gold_df = pd.read_json(f'{path_mcl_wic}/{split}.{langs}.gold', orient='records')
                    df = df.merge(gold_df)
                    for n in (1, 2):
                        df[f'pos{n}'] = df.apply(lambda r: [(r[f'start{n}'], r[f'end{n}'])], axis=1)
                        df = df.drop(columns=[f'start{n}', f'end{n}'])
                    df = df.rename({'lemma': 'word', 'sentence1': 'sent1', 'sentence2': 'sent2'}, axis=1)
                    data[f'{split}_{langs}_{name_data}'] = df
                else:
                    splits_mcl_wic = set(os.listdir(path_mcl_wic)) & full_splits_mcl_wic  # set of available splits, all possible splits at <full_splits_mcl_wic>
                    for split_mcl_wic in splits_mcl_wic:
                        name_files = [name.split('.')[1] for name in os.listdir(f'{path_mcl_wic}/{split_mcl_wic}') if name.startswith(f'{split}.') and
                                    (name.endswith('.data') or name.endswith('.gold'))]    # оставляем 'lang1-lang2' из файлов вида <split>.{lang1-lang2}.[gold/data]
                        name_files = set(name_files)
                        for langs in name_files:
                            path_gold = f'{path_mcl_wic}/{split_mcl_wic}/{split}.{langs}.gold'
                            is_gold_exits = os.path.exists(path_gold)

                            df = pd.read_json(f'{path_mcl_wic}/{split_mcl_wic}/{split}.{langs}.data', orient='records')
                            if is_gold_exits:
                                gold_df = pd.read_json(path_gold, orient='records')
                                df = df.merge(gold_df)
                            for n in (1, 2):
                                if f'start{n}' in df:
                                    df[f'pos{n}'] = df.apply(lambda r: [(r[f'start{n}'], r[f'end{n}'])], axis=1)
                                    df = df.drop(columns=[f'start{n}', f'end{n}'])
                            df = df.rename({'lemma': 'word', 'sentence1': 'sent1', 'sentence2': 'sent2'}, axis=1)
                            data[f'{split}_{langs}_{name_data}'] = df
        elif name_data == 'rusemshift':
            all_splits = ['train_comb', 'train_1', 'train_2', 'dev_1', 'dev_2']
            path_rusemshift = datasets_dir + '/' + name_data
            for split in all_splits:
                df = pd.read_csv(f'{path_rusemshift}/{split}.tsv', sep='\t')
                for n in (1,2):
                    df[f'pos{n}'] = df[f'sent{n}'].apply(extract_spans)
                    df[f'sent{n}'] = df[f'sent{n}'].apply(lambda s: s.replace('<b><i>', '').replace('</i></b>', ''))
                df['pos'] = 'NOUN'
                if drop_blank_position:
                    df = df[(df.pos1.str.len()>0)&(df.pos2.str.len()>0)].reset_index(drop=True)
                data[f'{split}_{name_data}'] = df
        else:
            raise ValueError("Wrong name dataset! Use 'mcl-wic' or 'rusemshift'.")
    return data