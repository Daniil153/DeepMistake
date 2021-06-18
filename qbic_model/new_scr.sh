#!/bin/bash
#SBATCH --gres=gpu:1 --time=3-00:00:00
IN=$1
OUT=${IN}.conllu
python tokenizer_.py --path $IN --out_path $OUT
mv $OUT GramEval2020/data/test_private_data/
rm GramEval2020/data/test_private_data/GramEval_private_test_clear.conllu
cd GramEval2020/solution
python -m train.applier --model-name ru_bert_final_model --batch-size 16
