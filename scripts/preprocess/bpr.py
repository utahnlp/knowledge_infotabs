import datetime
import random
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import json
import pandas as pd
import os
import sys
import requests
import re
import csv
import numpy as np
import argparse
from bs4 import BeautifulSoup
from collections import OrderedDict
import inflect
inflect = inflect.engine()


def is_date(string):
    match = re.search(r'\d{4}-\d{2}-\d{2}', string)
    if match:
        date = datetime.datetime.strptime(match.group(), '%Y-%m-%d').date()
        return True
    else:
        return False


def write_csv(data, split):
    with open(os.path.join(args['save_dir'], split + '.tsv'), 'at') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(data)


def table_to_para(file_path, file, nlp, table_cat_dic):
    f = open(file_path + file, encoding="utf-8")
    data = json.load(f, strict="False")
    category = category_from_keys(table_cat_dic[file.strip(".json")])
    sentences = ""
    if(args["mode"]=="1"):
        if(category):
            if(category != "default"):
                sentences += data["title"][0] + " is a " + category + ". " +"\n"
        else:
            actual_cat = table_cat_dic[file.strip(".json")]
            sentences += data["title"][0] + " is a " + actual_cat.lower() + ". " + "\n"
            category = "default"
        cnt = 0
        for j in data:
            if(j != "title"):
                txt = nlp(data[j][0])
                NER_value = ""
                for l in txt.ents:
                    NER_value = l.label_
                    break
                sent = template(category, NER_value, j.lower(),
                                data[j], data["title"][0])
                if(cnt != 0):
                    sentences += "\n" + sent
                else:
                    sentences += sent
                    cnt = 1
    else:
        if(category):
            if(category != "default"):
                sentences += data["title"][0] + " is a " + category + ". " 
        else:
            actual_cat = table_cat_dic[file.strip(".json")]
            sentences += data["title"][0] + " is a " + actual_cat.lower() + ". " 
            category = "default"
        cnt = 0
        for j in data:
            if(j != "title"):
                txt = nlp(data[j][0])
                NER_value = ""
                for l in txt.ents:
                    NER_value = l.label_
                    break
                sent = template(category, NER_value, j.lower(),
                                data[j], data["title"][0])
                if(cnt != 0):
                    sentences += sent + " "
                else:
                    sentences += sent + " "
                    cnt = 1
                    
    return sentences


def category_from_keys(category_name):
    catlower = category_name.lower()
    if catlower in [
        "food&drink",
        "album",
        "book",
        "wineyard",
        "sports event",
        "planet",
        "sport",
        "organization",
        "person",
        "musician",
        "animal",
        "painting",
        "country",
        "game",
        "movie",
        "city",
            "food"]:
        flag = 1
        return catlower
    else:
        flag = 1
        catlower = "default"
        return catlower


def template(cat, NER_va, key, value, title):
    sentence = ""
    if(cat in ["default", "food&drink", "album", "food"]):
        if(NER_va == "DATE"):
            sentence = sentence + title + " was " + key + " on "
        elif(key.lower() in ['olympic', 'paralympic']):
            if(value[0].lower() == 'yes'):
                sentence = sentence + title + " has " + key + ". "
                return sentence
            elif(value[0].lower() == 'no'):
                sentence = sentence + title + " does not have " + key + ". "
                return sentence
            else:
                if(len(value) == 1):
                    sentence = sentence + "The " + key + " of " + title + " is "
                else:
                    sentence = sentence + "The " + key + " of " + title + " are "
        elif(key.lower() in ['contact', 'mixed gender']):
            if(value[0].lower() == 'yes'):
                sentence = sentence + title + " is a " + key + " sport. "
                return sentence
            elif(value[0].lower() == 'no'):
                sentence = sentence + title + " is not a " + key + ". "
                return sentence
            else:
                if(len(value) == 1):
                    sentence = sentence + "The " + key + " of " + title + " is "
                else:
                    sentence = sentence + "The " + key + " of " + title + " are "
        elif(key.lower() in ['transfers', 'disabled access']):
            if(value[0].lower() == 'yes'):
                sentence = sentence + title + " has " + key + ". "
                return sentence
            elif(value[0].lower() == 'no'):
                sentence = sentence + title + " does not have " + key + ". "
                return sentence
            else:
                if(len(value) == 1):
                    sentence = sentence + "The " + key + " of " + title + " is "
                else:
                    sentence = sentence + "The " + key + " of " + title + " are "
        elif(key.lower() == 'single'):
            if(value[0].lower() == 'yes'):
                sentence = sentence + title + " is a " + key + ". "
                return sentence
            elif(value[0].lower() == 'no'):
                sentence = sentence + title + " is not a " + key + ". "
                return sentence
            else:
                if(len(value) == 1):
                    sentence = sentence + "The " + key + " of " + title + " is "
                else:
                    sentence = sentence + "The " + key + " of " + title + " are "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "book"):
        if(key == "schedule"):
            sentence = sentence + title + " has a " + \
                value[0] + " " + key + ". "
            return sentence
        elif(key == "volumes" and NER_va == "CARDINAL"):
            sentence = sentence + "The number of " + key + " of " + title + " are "
        elif(key[-2::] == "by"):
            sentence = sentence + title + " was " + key + " "
        elif(NER_va == "DATE"):
            if(key in ["publication date", "original run"]):
                sentence = sentence + "The " + key + " of " + title + " was on "
            else:
                sentence = sentence + title + " was " + key + " on "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat in ["wineyard", "sports event", "planet"]):
        if(len(value) == 1):
            sentence = sentence + "The " + key + " of " + title + " is "
        else:
            sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "sport"):
        if(NER_va == "CARDINAL"):
            sentence = sentence + "The number of " + key + " in " + title + " are "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "organization"):
        if(NER_va == "DATE"):
            if(key == "formation"):
                sentence = sentence + "The " + key + " of " + title + " was on "
            elif(key == "founded"):
                sentence = sentence + title + " was " + key + " on "
            else:
                sentence = sentence + title + " was " + key + " on "
        elif(NER_va == "CARDINAL"):
            sentence = sentence + "The number of " + key + " of " + title + " is "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat in ["person", "musician", "animal"]):
        if(NER_va == "DATE"):
            if(key in ["born", "foaled"]):
                sentence = sentence + title + " was " + key + " on "
            elif(key == "died"):
                sentence = sentence + title + " " + key + " on "
            else:
                sentence = sentence + "The " + key + " of " + title + " was on "
        elif(NER_va == "CARDINAL" and key == "children"):
            sentence = sentence + "The number of " + key + " of " + title + " are "
        else:
            if(key == "known for"):
                sentence = sentence + title + " was " + key + " "
            elif(key == "also known as"):
                sentence = sentence + title + " was " + key + " "
            elif(key == "born" and NER_va == "PERSON"):
                sentence = sentence + title + " was " + key + " as "
            else:
                if(len(value) == 1):
                    sentence = sentence + "The " + key + " of " + title + " is "
                else:
                    sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "painting"):
        if(key == "also known as"):
            sentence = sentence + title + " is " + key + " "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "country"):
        if(NER_va in ["DATE", "CARDINAL"]):
            sentence = sentence + "The " + key + " was on "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "game"):
        if(NER_va == "CARDINAL"):
            sentence = sentence + "The number of " + key + " of " + title + " are "
        elif(value[0] == "yes"):
            sentence = sentence + title + " is based on " + key + ". "
            return sentence
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "movie"):
        if(key[-2::] in ["by", "on"] or key == "starring"):
            sentence = sentence + title + " was " + key + " "
        elif(key == "box office"):
            sentence = sentence + "In the " + key + ", " + title + " made "
        elif(key == "cinematography"):
            sentence = sentence + "The " + key + " of " + title + " was by "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    elif(cat == "city"):
        if(NER_va == "DATE"):
            sentence = sentence + title + " was " + key + " on "
        elif(key in ['capital', 'legal deposit'] and value[0].lower() in ['yes', 'no']):
            if(value[0].lower() == 'yes'):
                sentence = sentence + title + " has a " + key + ". "
                return sentence
            elif(value[0].lower() == 'no'):
                sentence = sentence + title + " does not have a " + key + ". "
                return sentence
            else:
                if(len(value) == 1):
                    sentence = sentence + "The " + key + " of " + title + " is "
                else:
                    sentence = sentence + "The " + key + " of " + title + " are "
        elif(key in ["urban", "total", "metro", "city", "land", "water"]):
            sentence = sentence + "The " + key + " area of " + title + " is "
        else:
            if(len(value) == 1):
                sentence = sentence + "The " + key + " of " + title + " is "
            else:
                sentence = sentence + "The " + key + " of " + title + " are "
        for val in value:
            if(val == value[len(value) - 1]):
                sentence += val
            else:
                sentence += val + ', '
        sentence += ". "
    return sentence


def table_to_path_all(f_path, f_write_path, cat_path):
    nlp = en_core_web_sm.load()
    table_cat = pd.read_csv(os.path.join(
        cat_path, "table_categories.tsv"), sep='\t')
    for file in os.listdir(f_path):
        file_path = os.path.join(f_path, file)
        para, cat = table_to_para(file_path, file, nlp, table_cat)
        file_write = open(os.path.join(f_write_path, file[:-5:] + ".txt"), "w")
        file_write.writelines(para)
        file_write.close()


def config(parser):
    parser.add_argument(
        '--json_dir', default="./../data/tables/json/", type=str)
    parser.add_argument(
        '--mode', default="0", type=str)
    parser.add_argument(
        '--map', default="mnli", type=str)
    parser.add_argument(
        '--data_dir', default="./../data/maindata/", type=str)
    parser.add_argument(
        '--save_dir', default="./../../temp/bpr/", type=str)
    parser.add_argument('--cat_dir', default="./../data/tables/", type=str)
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
    #parser.add_argument('--rand_prem', default=0, type=int)
    #parser.add_argument('--multi_gpu_on', action='store_true')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = vars(parser.parse_args())
    nlp = en_core_web_sm.load()
    table_idtocat = {}
    para = {}
    table_cat = pd.read_csv(os.path.join(
        args['cat_dir'], "table_categories.tsv"), sep='\t')
    for index, row in table_cat.iterrows():
        table_idtocat[row['table_id']] = row['category']
    for table_id in table_idtocat.keys():
        para[table_id] = table_to_para(
            args['json_dir'],
            str(table_id) + ".json",
            nlp,
            table_idtocat)
    for split in args["splits"]:
        data = pd.read_csv(
            os.path.join(
                args['data_dir'],
                "infotabs_" +
                split +
                ".tsv"),
            sep="\t")
        with open(os.path.join(args['save_dir'], split + ".tsv"), 'wt') as out:
            writer = csv.writer(out, delimiter='\t')
            writer.writerow(["index", "table_id", "annotator_id",
                             "premise", "hypothesis", "label"])
        for index, row in data.iterrows():
            file_path = os.path.join(
                args['json_dir'], str(row['table_id']) + ".json")
            label = row["label"]
            if(args['map']=='mnli'):
                if row["label"] == "C":
                    label = 0
                if row["label"] == "N":
                    label = 1
                if row["label"] == "E":
                    label = 2
            else:
                if row["label"] == "E":
                    label = 0
                if row["label"] == "N":
                    label = 1
                if row["label"] == "C":
                    label = 2
            data = [index, row['table_id'], row['annotater_id'],
                    para[row['table_id']], row["hypothesis"], label]
            write_csv(data, split)
