#creating essential directories
mkdir ./../temp/data/opr
mkdir ./../temp/data/drr_ablation
mkdir ./../temp/data/kg_explicit_ablation

#bpr
python3 opr.py --json_dir ./../../data/tables/json/ --mode 0 --map mnli --data_dir ./../../data/maindata/ --save_dir ./../temp/data/opr/
#drr
python3 opr.py --json_dir ./../../data/tables/json/ --mode 1 --map mnli --data_dir ./../../data/maindata/ --save_dir ./../temp/data/drr_ablation/ 
python3 drr.py --json_dir ./../../data/tables/json/ --data_dir ./../temp/data/drr_ablation/ --save_dir ./../temp/data/drr_ablation/ --threshold 4 
#kg_explicit
python3 opr.py --json_dir ./../../data/tables/json/ --mode 1 --map mnli --data_dir ./../../data/maindata/ --save_dir ./../temp/data/kg_explicit_ablation/ 
python3 drr.py --json_dir ./../../data/tables/json/ --data_dir ./../temp/data/kg_explicit_ablation/ --save_dir ./../temp/data/kg_explicit_ablation/ --threshold 4 --sort 1
python3 kg_explicit.py --json_dir ./../../data/tables/json/ --data_dir ./../temp/data/kg_explicit_ablation/ --KG_dir ./../../data/kgdata/ --output_dir ./../temp/data/kg_explicit_ablation --kg_threshold 4 --order end
