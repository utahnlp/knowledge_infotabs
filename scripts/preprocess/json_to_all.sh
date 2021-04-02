#creating essential directories
mkdir ./../temp/data/bpr
mkdir ./../temp/data/drr
mkdir ./../temp/data/kg_explicit

#bpr
python3 bpr.py --json_dir ./../../data/tables/json/ --mode 0 --map mnli --data_dir ./../../data/maindata/ --save_dir ./../temp/data/bpr/ --cat_dir ./../../data/tables/
#drr
python3 bpr.py --json_dir ./../../data/tables/json/ --mode 1 --map mnli --data_dir ./../../data/maindata/ --save_dir ./../temp/data/drr/ --cat_dir ./../../data/tables/
python3 drr.py --json_dir ./../../data/tables/json/ --data_dir ./../temp/data/drr/ --save_dir ./../temp/data/drr/ --threshold 4 
#kg_explicit
python3 bpr.py --json_dir ./../../data/tables/json/ --mode 1 --map mnli --data_dir ./../../data/maindata/ --save_dir ./../temp/data/kg_explicit/ --cat_dir ./../../data/tables/
python3 drr.py --json_dir ./../../data/tables/json/ --data_dir ./../temp/data/kg_explicit/ --save_dir ./../temp/data/kg_explicit/ --threshold 4 --sort 1
python3 kg_explicit.py --json_dir ./../../data/tables/json/ --data_dir ./../temp/data/kg_explicit/ --KG_dir ./../../data/kgdata/ --output_dir ./../temp/data/kg_explicit --kg_threshold 4 --order end
