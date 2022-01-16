CUDA_VISIBLE_DEVICES=$1 python dice_finetune_on_drop.py   --do_train   --do_eval  --examples_n_features_dir ./data/examples_n_features/ --train_batch_size 15 --mlm_batch_size -1 --eval_batch_size 200 --learning_rate 3e-5  --max_seq_length 512 --num_train_epochs 50.0 --warmup_proportion 0.1 --init_weights_dir out_syntext_and_numeric_finetune_numeric --output_dir out_drop_finetune_syntext_numeric_ncmask_nodecnum --num_train_samples -1 --log_eval_step -1 --surface_form localattn --percent all --min_bound 0 --max_bound 9
