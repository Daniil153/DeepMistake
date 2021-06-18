import pandas as pd
import json
import os
import fire

def tsv_to_json(path_to_combine):
    if 'gold' not in os.listdir():
        os.mkdir('gold')
    for i in os.listdir(path_to_combine):
        temp = '_'.join(i.split('_')[:2]) + '.' + i.split('_')[2]
        df = pd.read_csv(path_to_combine + i, sep='\t')
        df['pos1'] = df.apply(lambda r: eval(r.pos1) if type(r.pos1) == type('str') else 0, axis=1)
        df['pos2'] = df.apply(lambda r: eval(r.pos2) if type(r.pos2) == type('str') else 0, axis=1)
        df = df[df.pos1 != 0]
        df = df[df.pos2 != 0]
        json_file = [
            {'id': f'{temp}.scd.{row[0]}', 'tag': 'T' if float(row[1]['mean']) > 2 else 'F', 'score': row[1]['mean']} for row in df.iterrows()]
        file_out = 'gold/' + temp + '.gold'
        f = open(file_out, 'w')
        json.dump(json_file, f, indent=4)



if __name__ == '__main__':
    fire.Fire(tsv_to_json)
