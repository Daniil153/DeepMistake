#SBATCH --gres=gpu:1
ckpt_path=$(dirname $1)
python run_model.py --do_eval --ckpt_path $ckpt_path --eval_input_dir ../pairs/combine/sampled_data --eval_output_dir rusemshift_predictions/ ${@:2}
