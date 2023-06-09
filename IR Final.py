import os
import string
import re
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk
import itertools
import math
import operator
import sys
from statistics import mean
import ir_datasets
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
#from stemming.porter2 import stem

from fastapi import FastAPI
###########################################################################################
app = FastAPI()

dataset = ir_datasets.load("antique/test")


#########################remove punctuation ###################################
def tokenization(strin):
    word_tokens = []
    for s in strin:
        translator = str.maketrans('', '', string.punctuation)
        modified_string = s.translate(translator)
        modified_string = ''.join(
            [i for i in modified_string if not i.isdigit()])
        word_tokens.append(word_tokenize(modified_string))
    return (word_tokens)


# REMOVE STOP WORDS ##################################333
def remove_stop_words(word_tokens):

    stop_words = list(stopwords.words('english'))
    filtered_sentence = []
    for w in word_tokens:
        for term in w:
            if not term.lower() in stop_words and not term.isdigit():
                filtered_sentence.append(term)
    return(filtered_sentence)
######################pos_tagge#########################################


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    ###########################LEMMATIZING################################################


def lemmitizing(wordnet_tagged):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag).lower())
    return(lemmatized_sentence)


##############################STEMMING#######################################

def stemming(lemmatized_sentence):
    stemed = []
    ps = PorterStemmer()
    for word in lemmatized_sentence:

        stemed.append(ps.stem(word))

    return(stemed)
    ############################INVERTED INDEX###########################################


def create_inverted_index(terms, docs_terms):
    inverted_index = defaultdict(list)
    for term in terms:

        for id, doc in docs_terms.items():

            for d in doc:

                if term == d:

                    inverted_index[term].append(id)
    return (inverted_index)


# ADD DOCS TO SET################################################3

def add_to_set(inverted_index):
    query_docs = set()
    for doc, terms in inverted_index.items():
        for term in terms:
            query_docs.add(term)
    return (query_docs)
    #################################PRINT INVERTED INDEX######################################


def print_inverted_index(inverted_index):
    for term, docs in inverted_index.items():
        print(term, ":", docs, "\n")
# 3


def dummy(doc):
    return doc
#####################################Count Vectorizer###########################################


def count_vector(stem_sentence):
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(stem_sentence)
    v = vectorizer.get_feature_names_out()
    return(v)
##################################TF IDF VECTOR##########################


def tf_idf(v, text1, text2):
    vectorize = TfidfVectorizer(
        analyzer="word", tokenizer=dummy, preprocessor=dummy, token_pattern=None, vocabulary=v)

    tfidf = vectorize.fit(text1)
    vector = tfidf.transform(text2)
    #df = pd.DataFrame(vector.toarray(), columns=vectorize.get_feature_names_out(), index=doc_idx)
    # print(df)
    return (tfidf)
#################################cosine_similarity###################################


def cosine_sim(tfidf1, tfidf2):
    cos_sim = cosine_similarity(tfidf1, tfidf2)
    return(cos_sim)

# 3


def create_terms_index(doc_idx, text):
    docs_index = defaultdict(list)
    for i in range(len(text)):

        for j in range(len(doc_idx)):
            if i == j:

                docs_index[doc_idx[j]].append(text[i])
    return (docs_index)
# 3


def get_docs(inverted_index, query):
    docs_q = set()
    for term, docs in inverted_index.items():
        if term in query:
            docs_q.update(docs)
    return docs_q
#########################################################################


def evaluation(df_sim, query_id):

    for qrel in dataset.qrels_iter():

        if qrel.query_id == query_id:
            print("qrel  ", qrel.query_id)
            if qrel.doc_id in df_sim.index:
                print(df_sim.loc[qrel.doc_id], qrel.relevance)
#######################################################################


################################### MAIN #################################
@app.get("/")
def reprocessing_docs(dataset_num):

    text = []
    doc_idx = []
    idxs = []
    prosseced_docs = []
    prossecing_text = defaultdict(list)

    for doc in dataset.docs_iter()[:100]:
        text.append(doc.text)
        doc_idx.append(doc.doc_id)
        idxs.append(doc.doc_id)

    terms_index = create_terms_index(doc_idx, text)

    for doc_idx, text in terms_index.items():
        tokenized_text = tokenization(text)
        filtered = remove_stop_words(tokenized_text)
        pos_tagged = nltk.pos_tag((filtered))
        wordnet_tagged = list(
            map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        lemmatized_text = lemmitizing(wordnet_tagged)
        stemmed_text = stemming(lemmatized_text)
        prosseced_docs.append(stemmed_text)
        prossecing_text[doc_idx] = stemmed_text

        set_terms = add_to_set(prossecing_text)

        inverted_index = create_inverted_index(set_terms, prossecing_text)
    return{"inverted_index": inverted_index, "prosseced_docs": prosseced_docs, "doc_idxs": idxs}


@app.get("/docs_tfidf")
def docs_tfidf(prosseced_docs, idxs):
    vectorizer = CountVectorizer(
        analyzer="word", tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    count_v = vectorizer.fit_transform(prosseced_docs)
    v = vectorizer.get_feature_names_out()
    vectorize = TfidfVectorizer(
        analyzer="word", tokenizer=dummy, preprocessor=dummy, token_pattern=None, vocabulary=v)
    tfidf = vectorize.fit(prosseced_docs)
    vector1 = tfidf.transform(prosseced_docs)
    vector1.shape
    df = pd.DataFrame(vector1.toarray(), index=idxs, columns=v)
    return{"df": df, "doc_tfidf_vector": tfidf}


@app.post("/query_process/{query}")
def query_processing(query, tfidf, inverted_index, df):
    que = []
    que_idx = []
    for query in dataset.queries_iter()[:1]:
        que.append(query.text)
        que_idx.append(query.query_id)
        tokenized_query = tokenization(que)
        filtered_query = remove_stop_words(tokenized_query)
        pos_tagged = nltk.pos_tag((filtered_query))
        wordnet_tagged = list(
            map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        lemmatized_query = lemmitizing(wordnet_tagged)
        str = []
        str2 = ''
        for l in lemmatized_query:
            str2 += l
            str2 += ' '
            str.append(str2)
        tfidf_q = tfidf.transform(str)
        tfidf_arr = tfidf_q
        df_q = pd.DataFrame(tfidf_arr.toarray())
        query_docs = get_docs(inverted_index, lemmatized_query)
        q_s = []
        for i in query_docs:
            q_s.append(i)
    dfidf_for_cosine = df.loc[q_s]
# print(dfidf_for_cosine)
    sim = cosine_similarity(dfidf_for_cosine, df_q)
    df_sim = pd.DataFrame(sim, index=q_s)
    print("df_sim", df_sim)
    evaluation(df_sim, que_idx)
