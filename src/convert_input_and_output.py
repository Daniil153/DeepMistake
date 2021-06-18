import pandas as pd
import numpy as np
import json

def convert_tsv_to_json(file):
    df = pd.read_csv(file)
    json_file = [{'id': row[0], 'lemma': row[1].word, 'pos': 'NOUN', 'sentence1': row[1].sent1,
                  'sentence2': row[1].sent2, 'start1': row[1].pos1[0], 'end1': row[1].pos1[1], 'start2': row[1].pos2[0],
                  'end2': row[1].pos2[1]} for row in df.iterrows()]
    file_out = file[:-4]
    f = open(file_out + '.json')
    json.dump(json_file, f)

def convert_ans_json_to_tsv(file):
    f = open('dev.ru-ru.data', 'r', encoding='utf-8')
    anses = json.load(f)