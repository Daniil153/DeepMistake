#!/bin/bash
ckpt_path=$(dirname $1)
mkdir for_gold
cp -r rushiftEval/*rusemshift for_gold/
python src/tsv_to_json_gold.py --path_to_combine for_gold/
python src/sample_scr.py --path_to_test rushiftEval/Evaluation_words/ --path_to_dev rushiftEval/RuSemShift_instances/ --path_to_sampled pairs/ --path_to_combine pairs/combine/ --in_words_dev1 rushiftEval/testset1.tsv --in_words_dev2 rushiftEval/testset2.tsv --delete_short_and_long False --delete_first_and_last_position False --n_pairs 100
bash mcl-wic/run_wic_model.sh ckpt_path ${@:2}
mkdir sampled_data
mkdir sampled_data/score
mkdir sampled_data/ans
mv mcl-wic/preds-rusemshift/rusemshift_trained_xlmrlarge-dmn/lr-2e-5-symmetric-true-lrs-linear_warmup-pool-mean-tembs-dist_l1ndotn-bn-1-linhead-True/*.scores sampled_data/score/
mv mcl-wic/preds-rusemshift/rusemshift_trained_xlmrlarge-dmn/lr-2e-5-symmetric-true-lrs-linear_warmup-pool-mean-tembs-dist_l1ndotn-bn-1-linhead-True/* sampled_data/ans/ 
python src/constr.py
python src/mean_method.py
python src/statistics_method.py
python src/iso_method.py
