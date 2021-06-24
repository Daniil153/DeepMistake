#!/bin/bash
mkdir -p for_gold
cp -r rushiftEval/*rusemshift for_gold/
python src/tsv_to_json_gold.py --path_to_combine for_gold/
python src/sample_scr.py --path_to_test rushiftEval/Evaluation_words/ --path_to_dev rushiftEval/RuSemShift_instances/ --path_to_sampled pairs/ --path_to_combine pairs/combine/ --in_words_dev1 rushiftEval/testset1.tsv --in_words_dev2 rushiftEval/testset2.tsv --delete_short_and_long True --delete_first_and_last_position True --n_pairs 100
echo "sample comlete"
cd mcl-wic
python run_model.py --do_eval --ckpt_path ../first_concat --eval_input_dir ../pairs/combine/sampled_data --eval_output_dir rusemshift_predictions/ --output_dir ../first_concat --loss mse_loss --pool_type first --symmetric true --train_scd 
cd ..
mkdir -p sampled_data
mkdir -p sampled_data/score
mkdir -p sampled_data/ans
mv first_concat/rusemshift_predictions/*.scores sampled_data/score/
mv first_concat/rusemshift_predictions/* sampled_data/ans/ 
python src/constr.py --type mean --model_name first_concat
python src/mean_method.py --model_name first_concat
python src/statistics_method.py --model_name first_concat
python src/iso_method.py --model_name first_concat
