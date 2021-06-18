from utils import get_time_index, get_positions
from vocative import vocative_lemmatizer
from tqdm.notebook import tqdm
import pathlib
import os
from typing import List


def get_sentence_index_tsv_dataset(
    original_dataset_path: str,
    dataset_lemmas_path: str,
    path_for_results: str,
    all_target_words: List[str]
) -> None:
    """
    """
    f_lemmas = open(dataset_lemmas_path)
    f_original = open(original_dataset_path)
    f_original_sentences = list(f_original)

    if not os.path.exists(path_for_results):
        os.makedirs(path_for_results)


    time_index = get_time_index(original_dataset_path)
    for i, sent in enumerate(tqdm(f_lemmas.readlines())):
        for w in all_target_words:
            if w in sent.split():
                target_word_index = all_target_words.index(w)
                new_file_path = pathlib.PurePath(
                    path_for_results,
                    f'{target_word_index}_{time_index}.tsv'
                    )
                new_f = open(new_file_path, 'a+')
                original_sent_str = f_original_sentences[i]
                word_lemma_pairs = vocative_lemmatizer(original_sent_str)
                positions = get_positions(w, word_lemma_pairs, original_sent_str)
                if len(positions):
                    string_to_file = original_sent_str.rstrip('\n') + '\t' + str(positions) + '\n'
                    new_f.write(string_to_file)
