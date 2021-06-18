print('start')
import nltk
import sys

temp = sys.argv[2]
temp_out = sys.argv[4]
f = open(temp)
a = f.read()
a = a.split('\n')
a = [i for i in a if len(i) >= 3]

str_1 = ""
for i in a:
    nltk_tokens = nltk.word_tokenize(i)
    tokens = []
    for j in nltk_tokens:
        ww = j.split('-')
        ww = [p for p in ww if len(p) >= 1]
        tokens.extend(ww)
    for j in range(len(tokens)):
        str_1 += str(j + 1) + "\t" + tokens[j] + "\n"
    str_1 += '\n'
#new_path = 'tokenize_' + temp.split('/')[-1]
#new_path = new_path.replace('.txt', '.conllu')
ff = open(temp_out, 'w')
ff.write(str_1)
print('end')

