train_ckpt=nen-nen-weights
pool=mean
targ_emb=dist_l1ndotn
ft_loss=crossentropy_loss
OUTPUT_DIR=xlmr-large..data_train-wic..train_loss-crossentropy_loss..data_ft-rusemshift..ft_loss-${ft_loss}..targ_emb-${targ_emb}..bn-1..pool-${pool}..ckpt-${train_ckpt}

linhead=true
python mcl-wic/run_model.py --do_train --do_validation --data_dir mcl-wic/data_dumped_full/wic --output_dir $OUTPUT_DIR/train/ --gradient_accumulation_steps 16 \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm 1 --loss crossentropy_loss --linear_head $linhead \
	--num_train_epochs 50 --symmetric true --local_config_path mcl-wic/local_config.json \
	--model_name xlm-roberta-large


train_scd=--train_scd
if [ $ft_loss = 'crossentropy_loss' ]; then
	train_scd=''
fi
python mcl-wic/run_model.py --do_train --do_validation $train_scd --data_dir mcl-wic/data_dumped_full/rusemshift-data/mean --output_dir $OUTPUT_DIR/finetune/ --gradient_accumulation_steps 16 \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm 1 --loss $ft_loss --linear_head $linhead \
	--num_train_epochs 50  --symmetric true --save_by_score spearman.dev.scd_2.score --local_config_path mcl-wic/local_config.json \
	--model_name xlm-roberta-large --ckpt_path $OUTPUT_DIR/train/$train_ckpt/