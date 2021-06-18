import os
os.chdir('ruword2tags')
from ruword2tags import RuWord2Tags
os.chdir('../')

import pandas as pd
from tqdm import tqdm
import pathlib
from typing import List, Set, Dict, Tuple, Callable
import rutokenizer
import rupostagger
import rulemma
from utils import extract_target_words_as_df


lemmatizer = rulemma.Lemmatizer()
lemmatizer.load()

tokenizer = rutokenizer.Tokenizer()
tokenizer.load()

tagger = rupostagger.RuPosTagger()
tagger.load()


def vocative_lemmatizer(sent: str) -> List[Tuple[str, str]]:
    tokens = tokenizer.tokenize(sent)
    tags = tagger.tag(tokens)
    lemmas = lemmatizer.lemmatize(tags)

    lemmas_list = []
    for word, tags, lemma, *_ in lemmas:
        lemmas_list.append((word, lemma))
    return lemmas_list


def save_lemmatized_datasets(
    results_folder: str,
    lemmatizer: Callable, 
    path_time1=None,
    path_time2=None,
    path_time3=None,
) -> None:
    """
    This function is aimed to form lemmatized versions of files.
    Files contain lemmatized sentencies.
    :param results_folder: path to folder which can be used as storage for
    results.
    :param path_time1: file, which contain sentences from 1st time period.
    :param path_time2: file, which contain sentences from 2nd time period.
    :param path_time3: file, which contatin sentences from 3rd time period.
    """
    paths = [
             path_time1,
             path_time2,
             path_time3
    ]

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for i, path_file in enumerate(paths):
        if path_file is not None:
            print(f'-----Processing {path_file}-----')

            file_name = path_file.split('/')[-1]
            new_dataset_path = pathlib.PurePath(results_folder, 'lemmatized_' + file_name)

            f_1 = open(path_file)

            for i, original_sent in enumerate(tqdm(f_1.readlines())):
                new_sent_list = lemmatizer(original_sent)
                new_sent_list = [w_l_pair[1] for w_l_pair in new_sent_list]
                new_sent = ' '.join(new_sent_list) + '\n'
                f_2 = open(new_dataset_path, 'a+')
                f_2.write(new_sent)
                f_2.close()
