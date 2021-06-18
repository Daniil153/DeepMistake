import pymorphy2
import os
import pandas as pd
from tqdm.notebook import tqdm
import pathlib
from typing import List, Tuple


def get_variants(current_word: str, pos: str = 'NOUN') -> List[str]:
    """
    To get all variants of considered word.
    :param current_word: word which forms need to get
    """
    variants: List[str] = []
    morph = pymorphy2.MorphAnalyzer()
    words = morph.parse(current_word)
    i = 0
    for new_word in words:
        if pos in new_word.tag:
          i += 1
          break
    if i == 0:
        print(f'There are no variants for word {current_word}.')
        return []

    cases = (('nomn', 'именительный'),
            ('gent', 'родительный'),
            ('datv', 'дательный'),
            ('accs', 'винительный'),
            ('ablt', 'творительный'),
            ('loct', 'предложный'))
    numbers = (('sing', 'Единственное'), ('plur', 'Множественное'))
    variants.append(current_word)
    for i in numbers:
        for j in cases:
            variants.append(new_word.inflect({i[0], j[0]}).word)
    return list(set(variants))


def detect_last_line(path_to_folder: str, path_to_file: str) -> int:
    last_lines = []
    last_line_value = -1
    for each_file in tqdm(os.listdir(path_to_folder)):
        final_path = path_to_folder + each_file

        with open(final_path, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        
        last_lines.append(last_line)

    f = open(path_to_file)
    for i, j in enumerate(tqdm(f.readlines())):
      for last_line in last_lines:
        if last_line in j and i > last_line_value:
          last_line_value = i
          break
    return last_line_value


def extract_target_words_as_file(path: str) -> List[str]:
    list_words = []
    f = open(path)
    for line in f.readlines():
        word = line.split('\t')[0]
        if word not in list_words:
            list_words.append(word)
    return list_words


def extract_target_words_as_df(path: str) -> List[str]:
    df = pd.read_csv(path, sep='\t', header=None)
    df_words = df.iloc[:, 0]
    return list(df_words.unique())


def get_positions(
    target_word: str,
    word_lemma_pairs: Tuple[str, str],
    sentence: str
) -> List[Tuple[int, int]]:
    positions = []
    end = 0
    for each_pair in word_lemma_pairs:
        original_word = each_pair[0].rstrip('\n')
        lemma = each_pair[-1]
        if target_word == lemma:
            start = sentence.find(original_word, end)
            end = start + len(original_word)
            positions.append((start, end))
    return positions


def get_time_index(path_time_dataset: str) -> int:
    if 'pre-soviet.txt' in path_time_dataset:
        time_index = 1
    elif 'post-soviet.txt' in path_time_dataset:
        time_index = 3
    else:
        time_index = 2
    return time_index
