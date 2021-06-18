train_ckpt=accuracy.dev.en-en.score
OUTPUT_DIR=xlmr-large..data_train-wic..train_loss-crossentropy_loss..data_ft-rusemshift..ft_loss-mse_loss..targ_emb-concat..bn-0..pool-first..ckpt-${train_ckpt}
python mcl-wic/run_model.py --do_train --do_validation --data_dir mcl-wic/data_dumped_full/wic --output_dir $OUTPUT_DIR/train/ --gradient_accumulation_steps 16 \
	--pool_type first --target_embeddings concat --head_batchnorm 0 --loss crossentropy_loss --linear_head false \
	--num_train_epochs 50 --symmetric true --save_by_score $train_ckpt --local_config_path mcl-wic/local_config.json \
	--model_name xlm-roberta-large --eval_per_epoch 10

python mcl-wic/run_model.py --do_train --do_validation --train_scd --data_dir mcl-wic/data_dumped_full/rusemshift-data/mean --output_dir $OUTPUT_DIR/finetune/ --gradient_accumulation_steps 16 \
	--pool_type first --target_embeddings concat --head_batchnorm 0 --loss mse_loss --linear_head false \
	--num_train_epochs 30  --symmetric true --save_by_score spearman.dev.scd_2.score --local_config_path mcl-wic/local_config.json \
	--model_name xlm-roberta-large --eval_per_epoch 10 --ckpt_path $OUTPUT_DIR/train/$train_ckpt/