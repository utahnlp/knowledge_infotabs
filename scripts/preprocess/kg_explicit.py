import pandas as pd
import os
import json
import re
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import argparse


def config(parser):
    parser.add_argument(
        '--json_dir', default="./../data/tables/json/", type=str)
    parser.add_argument(
        '--data_dir', default="./../temp/data/drr/", type=str)
    parser.add_argument(
        '--splits',
        default=[
            "train",
            "dev",
            "test_alpha1",
            "test_alpha2",
            "test_alpha3",
        ],
        action='store', type=str, nargs='*')
    parser.add_argument(
        '--KG_dir', default="./../data/kg_explicit/", type=str)
    parser.add_argument(
        '--output_dir', default="./../temp/data/kg_explicit", type=str)
    parser.add_argument(
        '--kg_threshold', default=4, type=int)
    parser.add_argument(
        '--order', default="end", type=str)
    
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = vars(parser.parse_args())

    tokenizer = RegexpTokenizer(r'\w+')

    dir_ = args["data_dir"]

    for split in args["splits"]:
        # ONLY PARA AND KG
        pruning_df = pd.read_csv(
            os.path.join(
                dir_,
                split + ".tsv"),
            sep="\t")
        keys_KG_dir = args["KG_dir"]
        
        json_dir = args["json_dir"]
        output_file_dir = args["output_dir"]
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)

        out_file_name = split + '.tsv'

        Write_file = open(
            os.path.join(
                output_file_dir,
                out_file_name),
            "w")
        first_line = "index" + "\t" + "table_id" + "\t" + "annotator_id" + \
            "\t" + "premise" + "\t" + "hypothesis" + "\t" + "label"
        Write_file.write(first_line + "\n")
        for index in range(pruning_df.shape[0]):
            file_path = open(
                os.path.join(
                    json_dir,
                    pruning_df['table_id'][index] +
                    ".json"),
                encoding='utf-8')
            json_data = json.load(file_path)
            df_KG_keys = pd.read_csv(
                os.path.join(
                    keys_KG_dir,
                    pruning_df['table_id'][index] +
                    '_KG.tsv'),
                sep='\t')
            
            para = str(pruning_df['premise'][index])
            para_dup = para
            hypo = pruning_df['hypothesis'][index]

            wikipedia_key = defaultdict(lambda: [])
            wordnet_key = defaultdict(lambda: [])
            owlbot_key = defaultdict(lambda: [])

            for j in range(df_KG_keys.shape[0]):
                if(df_KG_keys['source'][j] == 'wikipedia'):
                    if(not isinstance(df_KG_keys['keyword'][j], float)):
                        wikipedia_key[df_KG_keys['key'][j].lower().strip()].append([df_KG_keys['keyword'][j].lower(
                        ).strip(), df_KG_keys['definition'][j], df_KG_keys['description'][j]])
                elif(df_KG_keys['source'][j] == 'wordnet'):
                    if(not isinstance(df_KG_keys['keyword'][j], float)):
                        wordnet_key[df_KG_keys['key'][j].lower().strip()].append([df_KG_keys['keyword'][j].lower(
                        ).strip(), df_KG_keys['definition'][j], df_KG_keys['description'][j]])
                elif(df_KG_keys['source'][j] == 'owlbot'):
                    if(not isinstance(df_KG_keys['keyword'][j], float)):
                        owlbot_key[df_KG_keys['key'][j].lower().strip()].append([df_KG_keys['keyword'][j].lower(
                        ).strip(), df_KG_keys['definition'][j], df_KG_keys['description'][j]])

            

            para_list = re.split(r"(?<!\..)[.?!]\s+", para)
            para_kg = []
            for i in range(len(para_list)):
              for data in json_data:
                if(data != "title"):
                    if(data.lower().strip() in str(para_list[i]).lower()):
                        words = tokenizer.tokenize(data.lower().strip())
                        words = [w for w in words]

                        if(len(words) > 1):
                            if(data.lower().strip() in wikipedia_key):
                                if(len(wikipedia_key[data.lower().strip()]) == 1):
                                    para_kg.append((str(wikipedia_key[data.lower().strip()][0][0]) + " is defined as " + re.split(
                                        r"(?<!\..)[.?!]\s+", str(wikipedia_key[data.lower().strip()][0][1]))[0] + " ",i))
                                else:
                                    if(data.lower().strip() in wordnet_key):
                                        for defn in wordnet_key[data.lower(
                                        ).strip()]:
                                            para_kg.append((str(defn[0]) + " is defined as " + re.split(
                                                r"(?<!\..)[.?!]\s+", str(defn[1]))[0] + " ",i))
                        else:
                            if(data.lower().strip() in wordnet_key):
                                for defn in wordnet_key[data.lower(
                                ).strip()]:
                                    para_kg.append((str(defn[0]) + " is defined as " + re.split(
                                        r"(?<!\..)[.?!]\s+", str(defn[1]))[0] + " ",i))
            
                '''for value in json_data[data]:
            words=tokenizer.tokenize(value.lower().strip())
            words = [w for w in words]
            if(value.lower().strip() in hypo.lower().strip() and value not in ['january','february','march','april','may','june','july','august','september','october','november','december'] ):

              if(len(words)>1):
                if(value.lower().strip() in wikipedia_value):
                  if(len(wikipedia_value[value.lower().strip()])==1):

                    para+=" VALUE:"+str(wikipedia_value[value.lower().strip()][0][0])+" is defined as "+re.split(r"(?<!\..)[.?!]\s+",str(wikipedia_value[value.lower().strip()][0][1]))[0]+"."
                  else:
                    if(value.lower().strip() in owlbot_value):
                      for defn in owlbot_value[value.lower().strip()]:
                        para+=" VALUE:"+str(defn[0])+" is defined as "+re.split(r"(?<!\..)[.?!]\s+",str(defn[1]))[0]+"."
              else:
                if(value.lower().strip() in owlbot_value):
                  for defn in owlbot_value[value.lower().strip()]:
                    para+=" VALUE:"+str(defn[0])+" is defined as "+re.split(r"(?<!\..)[.?!]\s+",str(defn[1]))[0]+"."'''
            
            if(args["order"]=="end"):
                for i in range(min(len(para_list),args['kg_threshold'])):
                  for tup in para_kg:
                    if(tup[1]==i):
                      para+= "KEY: "+tup[0].capitalize()+". "

            else:

                add=""
                for i in range(min(len(para_list),args['kg_threshold'])):
                  for tup in para_kg:
                    if(tup[1]==i):
                      add+= "KEY: "+tup[0].capitalize()+". "
                para=add+para
              
            '''for month in [
                'january',
                'february',
                'march',
                'april',
                'may',
                'june',
                'july',
                'august',
                'september',
                'october',
                'november',
                    'december']:
                if(month in str(para_dup).lower().strip()):
                    if(month.lower().strip() in owlbot_value):
                        para += " VALUE: " + str(owlbot_value[month][0][0]).capitalize() + " is " + re.split(
                            r"(?<!\..)[.?!]\s+", str(owlbot_value[month][0][1]))[0]
                        if(para[-1]!='.'):
                        	para+= ". "
                        else:
                        	para+= " "'''

            new_line = str(pruning_df['index'][index]) + "\t" + pruning_df['table_id'][index] + "\t" + pruning_df['annotator_id'][index] + \
                "\t"  + str(para) + "\t" + pruning_df['hypothesis'][index] + "\t" + str(pruning_df['label'][index]) + "\n"
            Write_file.write(new_line)

        Write_file.close()
