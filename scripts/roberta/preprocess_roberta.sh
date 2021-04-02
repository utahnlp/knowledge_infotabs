# bpr
python3 preprocess_roberta.py --max_len 512 --data_dir ./../../temp/data/ --in_dir bpr --out_dir ../processed/bpr --single_sentence 0
# drr
python3 preprocess_roberta.py --max_len 512 --data_dir ./../../temp/data/ --in_dir drr --out_dir ../processed/drr --single_sentence 0
# kg_explicit
python3 preprocess_roberta.py --max_len 512 --data_dir ./../../temp/data/ --in_dir kg_explicit --out_dir ../processed/kg_explicit --single_sentence 0
#opr
python3 preprocess_roberta.py --max_len 512 --data_dir ./../../temp/data/ --in_dir opr --out_dir ../processed/opr --single_sentence 0
# drr_ablation
python3 preprocess_roberta.py --max_len 512 --data_dir ./../../temp/data/ --in_dir drr_ablation --out_dir ../processed/drr_ablation --single_sentence 0
# kg_explicit
python3 preprocess_roberta.py --max_len 512 --data_dir ./../../temp/data/ --in_dir kg_explicit_ablation --out_dir ../processed/kg_explicit_ablation --single_sentence 0
