#To replicate the results in Table 2 of the paper, the following commands can be referred to: for predicting, set mode to test, model_name(refer to Readme.txt)
######################COMMAND TO TRAIN BPR######################

#seeds used to report result in paper 13, 15, 17 (average of these accuracies are reported)

python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/bpr/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/bpr/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "bpr/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 17 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "none"

######################COMMAND TO TRAIN +KG implicit######################

#seeds used to report result in paper 11, 13, 23 (average of these accuracies are reported)

python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/bpr/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/kg_implicit/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "kg_implicit/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 13 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "implicit"

######################COMMAND TO TRAIN +DRR######################

#seeds used to report result in paper 17, 13, 23 (average of these accuracies are reported)

python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/drr/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/drr/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "drr/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 17 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "implicit"

######################COMMAND TO TRAIN +KG explicit######################

#seeds used to report result in paper 17, 19, 23 (average of these accuracies are reported)

python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/kg_explicit/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/kg_explicit/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "kg_explicit/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 17 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "implicit"

#To replicate the results in Table 3 of the paper or the ablation results, the following commands can be referred to: for predicting, set mode to test, model_name(refer to Readme.txt)

######################COMMAND TO TRAIN +DRR######################
python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/drr_ablation/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/drr_ablation/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "drr_ablation/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 17 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "none"

######################COMMAND TO TRAIN KG explicit######################
python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/kg_explicit_ablation/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/kg_explicit_ablation/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "kg_explicit_ablation/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 17 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "none"

######################COMMAND TO TRAIN KG implicit######################
python3 roberta_classifier.py \
	--mode "train" \
	--epochs 15 \
	--batch_size 8 \
	--in_dir "./../temp/processed/opr/" \
	--embed_size 1024 \
	--model_dir "./../temp/models/kg_implicit_ablation/" \
	--model_name "" \
	--save_dir "./../temp/models/" \
	--save_folder "kg_implicit_ablation/" \
	--nooflabels 3 \
	--save_enable 0 \
	--seed 17 \
	--eval_splits dev test_alpha1 \
	--inoculate 0 \
	--parallel 0 \
	--kg "implicit"