import spacy
import time
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
from bert_serving.client import BertClient
import os
import pandas as pd
import json
import string
import re
import wptools
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
#import mxnet as mx
#from bert_embedding import BertEmbedding
from scipy import spatial
import csv
from nltk.stem.snowball import SnowballStemmer
import nltk
import requests
nltk.download('wordnet')
url = 'https://owlbot.info/api/v4/dictionary/'
headers = {'Authorization' : 'Token 73860a5b6688182276ab8c7e003b752a24cb1480'}

table_dir = './../../data/tables/json'
extra_kg_dir = './../../data/kgdata/'
cat_dir='./../../data/tables/'
if not os.path.exists(extra_kg_dir):
        os.makedirs(extra_kg_dir)
table_idtocat = {}
# decide on this
# defn_dic={}
# desc_dic={}
############
df = pd.read_csv("GenericsKB-Best.tsv", sep='\t')
genericsKB = defaultdict(list)
for i in range(df.shape[0]):
    genericsKB[df['TERM'][i]].append(df['GENERIC SENTENCE'][i])
err = 0
nlp = spacy.load("en_core_web_sm")
#ctx = mx.gpu(0)
#bert = BertEmbedding(ctx=ctx)
bc = BertClient()
stemmer = SnowballStemmer("english")
table_cat = pd.read_csv(os.path.join(cat_dir,"table_categories.tsv"), sep='\t')
for index, row in table_cat.iterrows():
    table_idtocat[row['table_id']] = row['category']
fl = 0


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


def key_to_sentence(category, key, values, title):
    key = key.strip()
    txt = nlp(values[0])
    NER_value = ""
    for l in txt.ents:
        NER_value = l.label_
        break
    sent = template(category, NER_value, key.lower(),
                    values, title)
    return sent


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
    elif(cat == "organisation"):
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


def preprocess_key(key):
    keyn = re.sub(r"[\(\[].*?[\)\]]", "", key)
    keyn = re.sub(r"\d", "", keyn)
    keyn = re.sub(r"\d", "", keyn)
    trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    keyn = keyn.translate(trans)
    keyn = keyn.lower().strip()
    return keyn


def process_extext(text):
    text = text.replace('**', '')
    text = text.replace('_', '')
    text = text.replace('\n', ' ')
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    text = " ".join(word for word in text.split())
    if("to:" in text):
        t = text.split('*')
        if(len(t) > 1):
            text = t[1]
    return text


def definition(key, values, title, status, file):
    k = preprocess_key(key)
    try:
        pg = wptools.page(k)
        pg.get_query(show=False)
        desc = ""
        if('description' in pg.data):
            desc = pg.data['description']
        if 'disambiguation' not in desc:
            define = process_extext(pg.data['extext'])
            # defn_dic[key]=define
            # desc_dic[key]=desc
            return [define], [desc], [k]
        else:
            links = pg.data['links']
            define, desc = disambiguate(k, values, title, links, file)
            return [define], [desc], [k]

    except BaseException:
        if(status == 0):
            new_key = preprocess_key_2(k)
            define, desc, w = definition(new_key, values, title, 1, file)
            return define, desc, w
        elif(status == 1):
            defn = []
            descr = []
            w = []
            for word in k.split():
                defin, desc, _ = definition(word, values, title, 2, file)
                defn += defin
                descr += desc
                w += [word]
            return defn, descr, w
        else:
            return ["-1"], ["-1"], ["-1"]


def definition_wordnet(key, values, title, file):
    # used for both genericsKB and wordnet
    k = preprocess_key(key)
    new_key = preprocess_key_3(k)
    defn = []
    w = []
    lemmas = []
    for word in new_key.split():
        l = wn.synsets(word)
        defin, lemma = disambiguate_wordnet_main(word, values, title, l, file)
        defn += [defin]
        w += [word]
        lemmas += [lemma]
    return defn, w, lemmas

def definition_owlbot(key, values, title, file):
    # used for both genericsKB and wordnet
    k = preprocess_key(key)
    new_key = preprocess_key_3(k)
    defn = []
    w = []
    try:
      response=requests.get(url+new_key, headers=headers)
      response_data = json.loads(response.text)
      if('definitions' in response_data):
        defin=disambiguate_owlbot(new_key, values, title, response_data['definitions'], file)
        return [defin], [new_key]
    except:
      try:
        for word in new_key.split():
          response=requests.get(url+word, headers=headers)
          response_data = json.loads(response.text)
          if('definitions' in response_data):
            defin = disambiguate_owlbot(word, values, title, response_data['definitions'], file)
          defn += [defin]
          w += [word]
          
      except:
        pass
      
    return defn, w

def disambiguate_owlbot(k, values, title, diff_or, file):
    flag = 0
    try:
        category = category_from_keys(table_idtocat[file.strip(".json")])
    except BaseException:
        flag = 1
        category = "default"
    bert_key_val = key_to_sentence(category, k, values, title)
    if(flag != 1):
        if(category != "default"):
            bert_key_val = title + " is a " + category + " and " + bert_key_val
        else:
            actual_cat = table_idtocat[file.strip(".json")]
            ac = actual_cat.lower()
            bert_key_val = title + " is a " + str(ac) + " and " + bert_key_val
    sentences = bert_key_val.split('.')[:1]
    defin = ""
    
    max_sim = 0
    it = 0
    sent = sentences
    diff=[]
    for dic in diff_or:
        diff.append(dic)
    for dic in diff:
        if('example' in dic and dic['example']!=None):
            sent += [dic['example']]
        else:
            sent += [dic['definition']]
        it += 1
        if(it > 5):
            break
        
    res = bc.encode(sent, show_tokens=True)
    result = res[0][0]
    key_pos = -1
    for i in range(len(res[1][0])):
        if k.lower() in res[1][0][i].lower():
            key_pos = i
            break
    for s in range(1, len(sent)):
        key_pos_sent = -1
        for i in range(len(res[1][s])):
            if k.lower() in res[1][s][i].lower():
                key_pos_sent = i
                break
        if(key_pos_sent != -1 and key_pos != -1):
            cosine_similarity = 1 - \
                spatial.distance.cosine(result[key_pos], res[0][s][key_pos_sent])

        else:
            nparr = np.array(res[0][s])
            resavg = np.sum(nparr, axis=0)
            resavg = resavg / (len(res[1][s]) - 2)
            nparrres = np.array(result)
            result_avg = np.sum(nparrres, axis=0)
            result_avg = result_avg / (len(res[1][0]) - 2)
            cosine_similarity = 1 - spatial.distance.cosine(result_avg, resavg)
        if(cosine_similarity > max_sim):
            max_sim = cosine_similarity
            defin = diff[s-1]['definition']

    return defin

def preprocess_key_3(key):
    doc = nlp(key)
    docn = ""
    for token in doc:
        if(token.is_stop == False):
            docn += token.text + " "
    docn = docn.strip()
    return docn


def disambiguate_wordnet_main(k, values, title, diff_or, file):
    flag = 0
    try:
        category = category_from_keys(table_idtocat[file.strip(".json")])
    except BaseException:
        flag = 1
        category = "default"
    bert_key_val = key_to_sentence(category, k, values, title)
    if(flag != 1):
        if(category != "default"):
            bert_key_val = title + " is a " + category + " and " + bert_key_val
        else:
            actual_cat = table_idtocat[file.strip(".json")]
            ac = actual_cat.lower()
            bert_key_val = title + " is a " + str(ac) + " and " + bert_key_val
    sentences = bert_key_val.split('.')[:1]
    defin = ""
    lemma = ""
    max_sim = 0
    it = 0
    sent = sentences
    diff=[]
    for text in diff_or:
        diff.append(text)
    for text in diff:
        if(len(text.examples())!=0):
            txt = text.examples()[0]
            sent += txt.split('.')[:1]
        else:
            txt = text.definition()
            sent += txt.split('.')[:1]

        it += 1
        if(it > 5):
            break
        
    res = bc.encode(sent, show_tokens=True)
    result = res[0][0]
    key_pos = -1
    for i in range(len(res[1][0])):
        if str(k).lower() in str(res[1][0][i]).lower():
            key_pos = i
            break
    for s in range(1, len(sent)):
        key_pos_sent = -1
        for i in range(len(res[1][s])):
            if str(k).lower() in str(res[1][s][i]).lower():
                key_pos_sent = i
                break
        if(key_pos_sent != -1 and key_pos != -1):
            cosine_similarity = 1 - \
                spatial.distance.cosine(result[key_pos], res[0][s][key_pos_sent])

        else:
            nparr = np.array(res[0][s])
            resavg = np.sum(nparr, axis=0)
            resavg = resavg / (len(res[1][s]) - 2)
            nparrres = np.array(result)
            result_avg = np.sum(nparrres, axis=0)
            result_avg = result_avg / (len(res[1][0]) - 2)
            cosine_similarity = 1 - spatial.distance.cosine(result_avg, resavg)
        if(cosine_similarity > max_sim):
            max_sim = cosine_similarity
            defin = diff[s-1].definition()
            lemma = ', '.join([str(lemma.name())
                               for lemma in diff[s-1].lemmas()])

    return defin, lemma


def disambiguate_wordnet(k, values, title, diff, file):
    flag = 0
    try:
        category = category_from_keys(table_idtocat[file.strip(".json")])
    except BaseException:
        flag = 1
        category = "default"
    bert_key_val = key_to_sentence(category, k, values, title)
    if(flag != 1):
        if(category != "default"):
            bert_key_val = title + " is a " + category + " and " + bert_key_val
        else:
            actual_cat = table_idtocat[file.strip(".json")]
            ac = actual_cat.lower()
            bert_key_val = title + " is a " + str(ac) + " and " + bert_key_val
    sentences = bert_key_val.split('.')[:1]
    defin = ""
    max_sim = 0
    it = 0
    sent = sentences
    for text in diff:
        sent += text.split('.')[:1]
        it += 1
        if(it > 5):
            break
    res = bc.encode(sent, show_tokens=True)
    result = res[0][0]
    key_pos = -1
    for i in range(len(res[1][0])):
        if str(k).lower() in str(res[1][0][i]).lower():
            key_pos = i
            break
    for s in range(1, len(sent)):
        key_pos_sent = -1
        for i in range(len(res[1][s])):
            if str(k).lower() in str(res[1][s][i]).lower():
                key_pos_sent = i
                break
        if(key_pos_sent != -1 and key_pos != -1):
            cosine_similarity = 1 - \
                spatial.distance.cosine(result[key_pos], res[0][s][key_pos_sent])

        else:
            nparr = np.array(res[0][s])
            resavg = np.sum(nparr, axis=0)
            resavg = resavg / (len(res[1][s]) - 2)
            nparrres = np.array(result)
            result_avg = np.sum(nparrres, axis=0)
            result_avg = result_avg / (len(res[1][0]) - 2)
            cosine_similarity = 1 - spatial.distance.cosine(result_avg, resavg)
        if(cosine_similarity > max_sim):
            max_sim = cosine_similarity
            defin = sent[s]

    return defin


def definition_genericsKB(key, values, title, file):
    k = preprocess_key(key)
    # diff = search_genericsKB(key, 0) should be below one
    diff = search_genericsKB(k, 0)
    defn = []
    w = []
    if(len(diff) == 0):
        new_key = preprocess_key_2(k)
        for word in new_key.split():
            diff = search_genericsKB(word, 1)
            if(len(diff) != 0):
                defin = disambiguate_wordnet(
                    new_key, values, title, diff, file)
                defin = defin.replace('isa', 'is a')
                defn += [defin]
                w += [word]
    else:
        if(len(diff) != 0):
            defin = disambiguate_wordnet(k, values, title, diff, file)
            defin = defin.replace('isa', 'is a')
            defn += [defin]
            w += [k]
    return defn, w


def search_genericsKB(key, status):
    if(status == 0):
        kc = ""
        for i in key.split():
            kc += i.capitalize()
        if(key in genericsKB):
            diff = genericsKB[key]
        elif(key.capitalize() in genericsKB):
            diff = genericsKB[key.capitalize()]
        elif(kc in genericsKB):
            diff = genericsKB[kc]
        else:
            diff = []
    else:
        if(key in genericsKB):
            diff = genericsKB[key]
        elif(key.capitalize() in genericsKB):
            diff = genericsKB[key.capitalize()]
        else:
            diff = []
        if(len(diff) == 0):
            k = stemmer.stem(key)
            if(k in genericsKB):
                diff = genericsKB[k]
            elif(k.capitalize() in genericsKB):
                diff = genericsKB[k.capitalize()]
            else:
                diff = []
            '''if(len(diff)==0):
        #can this be done faster?
        #new_df=df.loc[df['TERM'].str.startswith(k, na=False)]
        #diff=list(new_df[['GENERIC SENTENCE', 'SCORE']].itertuples(index=False, name=None))
        diff=[]
        for i in genericsKB:
          if(str(i).startswith(k) and len(str(i))<=len(k)+3):
            diff+=genericsKB[i]'''
    return diff


def preprocess_key_2(key):
    doc = nlp(key)
    docn = ""
    for token in doc:
        if(token.is_stop == False):
            docn += token.lemma_ + " "
    docn = docn.strip()
    return docn


def disambiguate(k, values, title, links, file):
    flag = 0
    try:
        category = category_from_keys(table_idtocat[file.strip(".json")])
    except BaseException:
        flag = 1
        category = "deafult"
    bert_key_val = key_to_sentence(category, k, values, title)
    print(bert_key_val)
    if(flag != 1):
        if(category != "default"):
            bert_key_val = title + " is a " + category + " and " + bert_key_val
        else:
            actual_cat = table_idtocat[file.strip(".json")]
            bert_key_val = title + " is a " + actual_cat.lower() + " and " + bert_key_val
    sentences = bert_key_val.split('.')[:1]
    defin = ""
    desc = ""
    max_sim = 0
    it = 0
    sent = sentences
    descarr = []
    for link in links:
        pg = wptools.page(link)
        pg.get_query(show=False)
        description = ""
        if('description' in pg.data):
            description = pg.data['description']
            descarr.append(description)
        else:
            descarr.append('NA')
        text = process_extext(pg.data['extext'])
        sent += text.split('.')[:1]
        it += 1
        if(it > 5):
            break
    res = bc.encode(sent, show_tokens=True)
    result = res[0][0]
    key_pos = -1
    for i in range(len(res[1][0])):
        if str(k).lower() in str(res[1][0][i]).lower():
            key_pos = i
            break
    for s in range(1, len(sent)):
        key_pos_sent = -1
        for i in range(len(res[1][s])):
            if str(k).lower() in str(res[1][s][i]).lower():
                key_pos_sent = i
                break
        if(key_pos_sent != -1 and key_pos != -1):
            cosine_similarity = 1 - \
                spatial.distance.cosine(result[key_pos], res[0][s][key_pos_sent])

        else:
            nparr = np.array(res[0][s])
            resavg = np.sum(nparr, axis=0)
            resavg = resavg / (len(res[1][s]) - 2)
            nparrres = np.array(result)
            result_avg = np.sum(nparrres, axis=0)
            result_avg = result_avg / (len(res[1][0]) - 2)
            cosine_similarity = 1 - spatial.distance.cosine(result_avg, resavg)
        if(cosine_similarity > max_sim):
            max_sim = cosine_similarity
            defin = sent[s]
            desc = descarr[s - 1]

    return defin, desc


# if very slow then check if already found
s = time.time()
for file in os.listdir(table_dir):
    f_path = os.path.join(table_dir, file)
    fil_p = open(f_path, encoding='utf-8')
    data = json.load(fil_p)
    with open(os.path.join(extra_kg_dir, file[:-5:] + '_KG.tsv'), 'wt') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(
            ["key", "keyword", "definition", "description", "source"])
        title = data["title"][0]
        for key in data:
            if(key.lower() != "title"):
                value = data[key]
                value = value[0:1:]
                # WIKIPEDIA
                print('wikipedia')

                defin, desc, w = definition(key, value, title, 0, file)
                for i in range(len(defin)):
                    if(defin[i] != "-1"):
                        defin[i] = defin[i].replace('\t', ' ')
                        desc[i] = desc[i].replace('\t', ' ')
                        desc[i] = desc[i].replace('\n', ' ')
                        w[i] = w[i].replace('\t', ' ')
                        writer.writerow(
                            [key, w[i], defin[i], desc[i], "wikipedia"])
                else:
                    err += 1

                # WORDNET
                print('wordnet')
                defin2, w2, lem = definition_wordnet(key, value, title, file)
                for i in range(len(defin2)):
                    writer.writerow([key, w2[i], defin2[i], lem[i], "wordnet"])

                print('owlbot')
                defin4, w4 = definition_owlbot(key, value, title, file)
                for i in range(len(defin4)):
                    writer.writerow([key, w4[i], defin4[i], "NA", "owlbot"])

                # GENERICSKB
                print('genericsKB')
                defin3, w3 = definition_genericsKB(key, value, title, file)
                for i in range(len(defin3)):
                    writer.writerow(
                        [key, w3[i], defin3[i], "NA", "GenericsKB"])

    fl += 1
    print(file, fl)
e = time.time()
print(e - s)
