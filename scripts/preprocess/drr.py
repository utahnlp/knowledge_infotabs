import json
import os
import math
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer  # for nltk word tokenization
from nltk.stem.wordnet import WordNetLemmatizer
import math
import re
import fasttext.util
import argparse
from collections import OrderedDict 

def Preprocess_QA_sentences(sentences, stop_word_flag):
    words = tokenizer.tokenize(sentences.lower())
    words = [lmtzr.lemmatize(w1) for w1 in words]
    if stop_word_flag == 1:
        words = [w for w in words if w not in stop_words]
    return words


def sent_Emb(ht_terms, embeddings_index, emb_size=300):
    HT_Matrix = np.empty((0, emb_size), float)
    tokens_not_found_embeddings = []
    tokens_embeddings_found = []
    for ht_term in ht_terms:
        coefs = np.asarray(
            embeddings_index.get_word_vector(ht_term),
            dtype='float32')
        b = np.linalg.norm(coefs, ord=2)
        if(b != 0):
            coefs = coefs / float(b)
            #np.linalg.norm(coefs, ord=2)
        
        HT_Matrix = np.append(HT_Matrix, np.array([coefs]), axis=0)
        tokens_embeddings_found.append(ht_term)
        '''HT_Matrix = np.append(HT_Matrix, np.array([np.asarray(embeddings_index.get_word_vector(ht_term))]), axis=0)
        tokens_embeddings_found.append(ht_term)
        except:
           tokens_not_found_embeddings.append(ht_term)'''
    return HT_Matrix, tokens_not_found_embeddings, tokens_embeddings_found


def compute_alignment_vector(
        Hypo_matrix,
        hypo_toks_nf,
        hypo_toks_found,
        table_matrix,
        table_toks_nf,
        threshold=0.90):
    table_matrix = table_matrix.transpose()
    Score = np.matmul(Hypo_matrix, table_matrix)
    Score = np.sort(Score, axis=1)
    max_score1 = Score[:, -1:]  # taking the highest element column
    max_score1 = np.asarray(max_score1).flatten()

    max_score1 = [1 if s1 >= threshold else 0 for s1 in max_score1]
    remaining_terms = []

    for i1, s1 in enumerate(max_score1):
        if s1 == 0:
            remaining_terms.append(hypo_toks_found[i1])

    for t1 in hypo_toks_nf:
        if t1 in table_toks_nf:
            max_score1.append(1)
        else:
            max_score1.append(0)
            remaining_terms.append(t1)

    return remaining_terms


def compute_alignment_score(
        Hypo_matrix,
        hypo_toks_nf,
        table_matrix,
        table_toks_nf):
    table_matrix = table_matrix.transpose()
    Score = np.matmul(Hypo_matrix, table_matrix)
    Score = np.sort(Score, axis=1)
    max_score1 = Score[:, -1:]  # taking 3 highest element columns
    max_score1 = np.asarray(max_score1).flatten()

    return np.sum(max_score1)


def get_alignment_justification(
        hypo_terms,
        table_k_v,
        title,
        embedding_index,
        emb_size,
        flag=1):
    Hypo_matrix, hypo_toks_nf, hypo_toks_found = sent_Emb(
        hypo_terms, embedding_index, emb_size)
    table_hyp_remaining_terms = {}
    Final_alignment_scores = []
    num_remaining_terms = []

    for ind, t_terms in enumerate(table_k_v):
        table_terms = Preprocess_QA_sentences(t_terms, 1)
        table_terms = list(set(table_terms) - set(title))
        table_matrix, table_toks_nf, table_toks_found = sent_Emb(
            table_terms, embedding_index, emb_size)
        table_hyp_remaining_terms.update({ind: compute_alignment_vector(
            Hypo_matrix, hypo_toks_nf, hypo_toks_found, table_matrix, table_toks_nf)})
        ind_score = compute_alignment_score(
            Hypo_matrix, table_toks_nf, table_matrix, table_toks_nf)
        num_remaining_terms.append(len(table_hyp_remaining_terms[ind]))
        Final_alignment_scores.append(ind_score)
    # the higher alignment score, the more similar it is
    Final_index = list(np.argsort(Final_alignment_scores)[::-1])
    if(flag == 0):
        return Final_index[0], table_hyp_remaining_terms[Final_index[0]
                                                         ], table_hyp_remaining_terms
    else:
        if(args["sort"]=="0"):
            final_list = {}
        else:
            final_list = OrderedDict()
        for i in range(min(len(Final_index), args["threshold"])):
            final_list[Final_index[i]] = Final_alignment_scores[Final_index[i]]

        return final_list, table_hyp_remaining_terms[Final_index[0]
                                                     ], table_hyp_remaining_terms


# Alignment over embeddings for sentence selection
def get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(
        hypo_text, table_k_v, title, embedding_index, emb_size=300):
    hypo_terms = Preprocess_QA_sentences(hypo_text, 1)
    title_terms = Preprocess_QA_sentences(title, 1)
    hypo_terms = list(set(hypo_terms) - set(title_terms))
    if(args["sort"]=="0"):
        indexes = set()
    else:
        indexes = []
    first_iteration_index1, remaining_toks1, remaining_toks2 = get_alignment_justification(
        hypo_terms, table_k_v, title_terms, embedding_index, emb_size, 1)
    for i in first_iteration_index1:
        if(args["sort"]=="0"):
            indexes.add(i)
        else:
            indexes.append(i)

    return indexes, first_iteration_index1

def config(parser):
    parser.add_argument(
        '--json_dir', default="./../data/tables/json/", type=str)
    parser.add_argument(
        '--data_dir', default="./../temp/data/drr/", type=str)
    parser.add_argument(
        '--save_dir', default="./../temp/data/drr/", type=str)
    parser.add_argument(
        '--sort', default="0", type=str)
    parser.add_argument(
        '--threshold', default=4, type=int)
    parser.add_argument(
        '--splits',
        default=[
            "train",
            "dev",
            "test_alpha1",
            "test_alpha2",
            "test_alpha3",
            ],
        action='store',
        type=str,
        nargs='*')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = vars(parser.parse_args())

    fasttext.util.download_model('en', if_exists='ignore')  # English
    lmtzr = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r"(?x)(?:[A-Za-z]\.)+| \w+(?:\w+)*")
    stop_words = [
        'i',
        'me',
        'my',
        'myself',
        'we',
        'our',
        'ours',
        'ourselves',
        'you',
        'your',
        'yours',
        'yourself',
        'yourselves',
        'he',
        'him',
        'his',
        'himself',
        'she',
        'her',
        'hers',
        'herself',
        'it',
        'its',
        'itself',
        'they',
        'them',
        'their',
        'theirs',
        'themselves',
        'what',
        'which',
        'who',
        'whom',
        'this',
        'that',
        'these',
        'those',
        'am',
        'is',
        'are',
        'was',
        'were',
        'be',
        'been',
        'being',
        'have',
        'has',
        'had',
        'having',
        'do',
        'does',
        'did',
        'doing',
        'a',
        'an',
        'the',
        'and',
        'but',
        'if',
        'or',
        'because',
        'as',
        'until',
        'while',
        'of',
        'at',
        'by',
        'for',
        'with',
        'about',
        'against',
        'between',
        'into',
        'through',
        'during',
        'before',
        'after',
        'above',
        'below',
        'to',
        'from',
        'up',
        'down',
        'in',
        'out',
        'on',
        'off',
        'over',
        'under',
        'again',
        'further',
        'then',
        'once',
        'here',
        'there',
        'when',
        'where',
        'why',
        'how',
        'all',
        'any',
        'both',
        'each',
        'few',
        'more',
        'most',
        'other',
        'some',
        'such',
        'no',
        'nor',
        'not',
        'only',
        'own',
        'same',
        'so',
        'than',
        'too',
        'very',
        's',
        't',
        'can',
        'will',
        'just',
        'don',
        'should',
        'now',
        'd',
        'll',
        'm',
        'o',
        're',
        've',
        'y',
        'ain',
        'aren',
        'couldn',
        'didn',
        'doesn',
        'hadn',
        'hasn',
        'haven',
        'isn',
        'ma',
        'mightn',
        'mustn',
        'needn',
        'shan',
        'shouldn',
        'wasn',
        'weren',
        'won',
        'wouldn']
    stop_words = [lmtzr.lemmatize(w1) for w1 in stop_words]
    stop_words = list(set(stop_words))

    ft = fasttext.load_model('./cc.en.300.bin')
    emb_size = 300
      # split_set = "dev"

    output_file_dir = args['save_dir']
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    for split in args['splits']:
        input_file_name = os.path.join(args['data_dir'], split + ".tsv")
        out_file_name = split + ".tsv"
        
        df = pd.read_csv(input_file_name, sep='\t')

        Write_file = open(
            os.path.join(
                output_file_dir,
                out_file_name),
            "w")
        first_line = "index" + "\t" + "table_id" + "\t" + "annotator_id" + \
            "\t" + "premise" + "\t" + "hypothesis" +  "\t" + "label"
        Write_file.write(first_line + "\n")
        json_dir = args["json_dir"]

        for index in range(df.shape[0]):
            file_path = open(
                os.path.join(
                    json_dir,
                    df['table_id'][index] +
                    ".json"),
                encoding='utf-8')
            json_data = json.load(file_path)

            table_k_v = []
            title = json_data['title'][0]

            para = df['premise'][index] + " "
            sentences = para.split("\n")
            for sent in sentences:
                if(sent != ''):
                    table_k_v.append(sent)
            hypo_text = df['hypothesis'][index]

            ind, scores = get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(
                hypo_text, table_k_v, title, ft, emb_size=emb_size)
            keys_vals = ""
            for i in ind:                
                keys_vals = keys_vals + " " + table_k_v[i] #+ "."

            # print (type(df['index'][index]),type(df['table_id'][index]),type(df['annotator_id'][index]),type(hypo_text),type(keys_vals),type(df['label'][index]))

            new_line = str(df['index'][index]) + "\t" + df['table_id'][index] + "\t" + \
                df['annotator_id'][index] + "\t" + keys_vals + "\t" + hypo_text +  "\t" + str(df['label'][index]) + "\n"
            Write_file.write(new_line)

        Write_file.close()
