#!/bin/bash
cd GramEval2020
pip install -r requirements.txt
pip install git+git://github.com/DanAnastasyev/allennlp.git
./download_model.sh ru_bert_final_model

