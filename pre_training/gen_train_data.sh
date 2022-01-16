python gen_train_data_MLM.py --train_corpus ./data/MLM_paras.jsonl --bert_model $1 --output_dir ./data/MLM_train_albert/ --do_lower_case --max_predictions_per_seq 65 --digitize
