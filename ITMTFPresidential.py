import pandas as pd
import xml.etree.ElementTree as ET
import glob
import os
import pickle
from gensim.utils import lemmatize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from scipy.stats import pearsonr
import numpy as np
import re 
import string
import math

def parse_nyt_corpus_for_gore_bush():

    root_dir = r'nyt_corpus/'
    gore_bush_nyt_ts = []
    gore_bush_unsplit_doc_list = []

    #each directory in the corpus directory is a month
    for month in os.listdir(root_dir):
        month_dir = root_dir + month + r'/' + month + r'/'
        #each directory in the month directory is a day
        for day in os.listdir(month_dir):
            day_dir = month_dir + day + r'/'
            day_files = glob.glob(day_dir + r'*.xml')
            date_timestamp = month + r'/' + day + r'/' + r'2000'
            #for each of the files in a day directory
            for doc in day_files:
                #parse the xml by first getting the element tree and its root
                xml_tree = ET.parse(doc)
                root = xml_tree.getroot()
                #build the document text by stringing together all <p></p> element values in <block class="full text></block>
                doc_txt = ''
                body = root.find('body')
                body_content = body.find('body.content')
                for block in body_content.findall('block'):
                    if block.get('class') == 'full_text':
                        p_txt = r' '
                        for paragraph in block.findall('p'):
                            p_txt = paragraph.text.strip() + r' '
                            #matches = [r' Gore ',r' Bush ']
                            #if any(x in p_txt for x in matches):
                            if re.search(r'\WGore\W|\WBush\W',p_txt):
                                doc_txt += p_txt
                                #if 'giuliani' in p_txt.lower():
                                #    print('giuliani docId: ' + str(len(gore_bush_nyt_ts)))
                if doc_txt != '' and doc_txt != ' ':
                    gore_bush_nyt_ts.append((doc_txt,date_timestamp))

    return gore_bush_nyt_ts

def build_datasets():

    president_norm_stock_df = pd.read_csv('president_normalized_prices_may_oct_2000.csv')
    president_norm_stock_ts = [tuple(x) for x in president_norm_stock_df.to_numpy()]
    gore_bush_nyt_ts = parse_nyt_corpus_for_gore_bush()

    return president_norm_stock_ts,gore_bush_nyt_ts


def load_data(rebuild=False):
    """
    returns two dicts of {t,d} tuples
    d = data
    t = timestamp (a string date representation in this case with format mm/dd/YYYY)
    
    First list is presidential normalized stock price time series data
    Second list is made up of a document time series where each document is the paragraph data from the NY Times relevant to either George Bush or Al Gore
    """
    if rebuild == False:
        try:
            president_norm_stock_ts = pickle.load(open('president_norm_stock_ts.pkl', 'rb'))
            print('president_norm_stock_ts loaded...')
            gore_bush_nyt_ts = pickle.load(open('gore_bush_nyt_ts.pkl', 'rb'))
            print('gore_bush_nyt_ts loaded...')
        except:
            president_norm_stock_ts,gore_bush_nyt_ts = build_datasets()
            pickle.dump(president_norm_stock_ts, open('president_norm_stock_ts.pkl', 'wb'))
            print('president_norm_stock_ts built...')
            pickle.dump(gore_bush_nyt_ts, open('gore_bush_nyt_ts.pkl', 'wb'))
            print('gore_bush_nyt_ts built...')
    else:
        president_norm_stock_ts,gore_bush_nyt_ts = build_datasets()
        pickle.dump(president_norm_stock_ts, open('president_norm_stock_ts.pkl', 'wb'))
        print('president_norm_stock_ts built...')
        pickle.dump(gore_bush_nyt_ts, open('gore_bush_nyt_ts.pkl', 'wb'))
        print('gore_bush_nyt_ts built...')

    return president_norm_stock_ts,gore_bush_nyt_ts

def get_stop_words():
    download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['mr','gore','al','george','w','bush','gores','bushs','said'])
    return stop_words

def process_doc(text, stop_words, w_lemmatizer = False):
    """Read in a text string, clean the words by removing stopwords, removing puncuation, and lemmatizing and return a list of words"""

    #also we don't need to POS tag here so split on the / and remove it
    if w_lemmatizer:
        doc = lemmatize(text,stopwords=stop_words)
        #if POS:
        #    doc = lemmatize(text,stopwords=stop_words)
        #else:
        #    doc = [wd.decode().split('/')[0] for wd in lemmatize(text,stopwords=stop_words)]
    else:
        punctuation_chars = re.escape(string.punctuation)
        text = re.sub(r'['+punctuation_chars+']', '', text)
        text = text.lower()
        doc = word_tokenize(text)
        doc = [word for word in doc if word not in stop_words]
        doc = [word for word in doc if word.isalpha()]
    return doc

def build_doc_list(list_dirty_docs,stop_words,w_lemmatizer = False):
    doc_list = []
    for doc in list_dirty_docs:
        new_doc = process_doc(doc,stop_words,w_lemmatizer)
        doc_list.append(new_doc)
    return doc_list

def clean_collection(list_dirty_docs,stop_words,rebuild=False,w_lemmatizer = False):

    if rebuild == False:
        try:
            cleaned_doc_list = pickle.load(open('cleaned_doc_list.pkl', 'rb'))
            print('cleaned_doc_list loaded...')
        except:
            cleaned_doc_list = build_doc_list(list_dirty_docs,stop_words,w_lemmatizer)
            pickle.dump(cleaned_doc_list, open('cleaned_doc_list.pkl', 'wb'))
            print('cleaned_doc_list built...')
    else:
        cleaned_doc_list = build_doc_list(list_dirty_docs,stop_words,w_lemmatizer)
        pickle.dump(cleaned_doc_list, open('cleaned_doc_list.pkl', 'wb'))
        print('cleaned_doc_list built...')

    return cleaned_doc_list

def build_gensim_corpus(list_of_docs,rebuild=False):

    if rebuild:
        gensim_dictionary = Dictionary(list_of_docs)
        gensim_corpus = [gensim_dictionary.doc2bow(text) for text in list_of_docs]
        pickle.dump(gensim_dictionary, open('gensim_dictionary.pkl', 'wb'))
        print('gensim_dictionary built...')
        pickle.dump(gensim_corpus, open('gensim_corpus.pkl', 'wb'))
        print('gensim_corpus built...')
    else:
        try:
            gensim_dictionary = pickle.load(open('gensim_dictionary.pkl', 'rb'))
            print('gensim_dictionary loaded...')
            gensim_corpus = pickle.load(open('gensim_corpus.pkl', 'rb'))
            print('gensim_corpus loaded...')
        except:
            gensim_dictionary = Dictionary(list_of_docs)
            gensim_corpus = [gensim_dictionary.doc2bow(text) for text in list_of_docs]
            pickle.dump(gensim_dictionary, open('gensim_dictionary.pkl', 'wb'))
            print('gensim_dictionary built...')
            pickle.dump(gensim_corpus, open('gensim_corpus.pkl', 'wb'))
            print('gensim_corpus built...')

    return gensim_dictionary,gensim_corpus

def build_topic_stream(lda_model,corpus,doc_ts,ts_tsID_map):
    """build a matrix where each row is a timestamp and each column is a topic
    the value is that topic's coverage for that day"""

    #first build a dict of {docId:timestampID}
    doc_ts_dict = {}

    for t in range(0,len(corpus)):
        doc_ts_dict[t] = ts_tsID_map[doc_ts[t][1]]

    #initialize the topic stream matrix
    topic_stream_matrix = np.zeros((lda_model.num_topics,len(ts_tsID_map)))
    
    #fill in the matrix for the day:topic
    #use the doc_ts_dict to get the row index from doc_ts_dict
    #column index is the first item in the topicID,topicDocCoverage from lda_model[corpus[docID]]
    for docID in range(0,len(corpus)):
        for topic in lda_model[corpus[docID]]:
            topicID = topic[0]
            topicDocCoverage = topic[1]
            topic_stream_matrix[topicID,doc_ts_dict[docID]] += topicDocCoverage

    return topic_stream_matrix

def find_causal_topics(num_topics,topic_causality_df,x_col,p_cutoff):

    causal_topics = []
    sum_causal_p_vals = 0
    for topicID in range(0,num_topics):
        pearson_res = pearsonr(topic_causality_df[x_col],topic_causality_df[topicID])
        p_val = pearson_res[1]
        #granger_result = grangercausalitytests(causal_df[[x_col,topicID]], maxlag=1)
        #p_val = granger_result[1][0]['params_ftest'][1]
        if p_val >= p_cutoff:
            causal_topics.append(topicID)
            sum_causal_p_vals += p_val
    if len(causal_topics) > 0:
        avg_causality_confidence = sum_causal_p_vals/(len(causal_topics))
    else:
        avg_causality_confidence = 0
    return causal_topics,avg_causality_confidence

def build_word_count_stream(corpus,dictionary,doc_ts,ts_tsID_map):
    """build a matrix where each row is a timestamp and each column is a topic
    the value is that topic's coverage for that day"""

    #first build a dict of {docId:timestampID}
    doc_ts_dict = {}

    for t in range(0,len(corpus)):
        doc_ts_dict[t] = ts_tsID_map[doc_ts[t][1]]

    #initialize the topic stream matrix
    wc_stream_matrix = np.zeros((len(dictionary),len(ts_tsID_map)))
    
    #fill in the matrix for the day:topic
    #use the doc_ts_dict to get the column index from doc_ts_dict
    #row index is the first item in the topicID,topicDocCoverage from corpus[docID]
    for docID in range(0,len(corpus)):
        for word in corpus[docID]:
            wordID = word[0]
            word_doc_cnt = word[1]
            wc_stream_matrix[wordID,doc_ts_dict[docID]] += word_doc_cnt

    return wc_stream_matrix

def build_word_impact_dict(num_words,word_causal_df,x_col):

    word_impact_dict = {}
    for wordID in range(0,num_words):
        pearson_res = pearsonr(word_causal_df[x_col],word_causal_df[wordID])
        impact = pearson_res[0]
        p_val = pearson_res[1]
        word_impact_dict[wordID] = (impact,p_val)
        #if p_val >= p_cutoff:
        #    word_impact_dict[wordID] = (impact,p_val)
    return word_impact_dict

def calc_prob_word_new_topic(cur_word_sig,sig_total,p_cutoff):
    prob = (cur_word_sig - p_cutoff)/sig_total
    return prob

def calc_new_word_prob_for_topic(old_topic_matrix_row,impact_words):

    total_sigs = np.sum(old_topic_matrix_row)

    for wordID in impact_words:
        old_prob = old_topic_matrix_row[wordID]
        new_prob = float(old_prob)/float(total_sigs)
        old_topic_matrix_row[wordID] = new_prob
    total = np.sum(old_topic_matrix_row)
    return old_topic_matrix_row

def calculate_topic_purity(num_pos_words,num_neg_words):
    p_prob = float(num_pos_words)/float((num_pos_words + num_neg_words))
    n_prob = float(num_neg_words)/float((num_pos_words + num_neg_words))
    topic_entropy = (p_prob * (math.log(p_prob,2))) + (n_prob * (math.log(n_prob,2)))
    topic_purity = 100 + (100 * topic_entropy)
    return topic_purity

def build_prior_matrix(word_impact_dict,causal_topics,term_topic_matrix,p_cutoff,prob_cutoff):

    #build a dict of topicID:[wordIDs with topic probability > prob_cutoff]
    topic_probM_wordID_dict = {}
    wordIDs = []
    for topicID in causal_topics:
        wordIDs_greater_than_probM_np = np.where(np.array(term_topic_matrix[topicID]) >= prob_cutoff)
        topic_probM_wordID_dict[topicID] = list(wordIDs_greater_than_probM_np[0])
        wordIDs.extend(topic_probM_wordID_dict[topicID])
    
    #unique_top_wordIDs = list(set(wordIDs))
    
    #A dict to know which column to place this word's prior probability in
    #unique_top_wordID_idx_dict = {}
    #for idx in range(0,len(unique_top_wordIDs)):
    #    unique_top_wordID_idx_dict[unique_top_wordIDs[idx]] = idx
    
    #initialize prior_matrix
    #word_sig_by_topic_matrix = []
    topic_purities = np.zeros((len(causal_topics)))
    tp_cnt = 0
    new_topic_cnt = len(term_topic_matrix)
    prior_matrix = np.zeros((new_topic_cnt,len(word_impact_dict)))
    for topicID in causal_topics:
        pos_impact_words = []
        pos_sig_total = 0
        neg_impact_words = []
        neg_sig_total = 0
        for wordID in topic_probM_wordID_dict[topicID]:
            impact = word_impact_dict[wordID][0]
            p_val = word_impact_dict[wordID][1]
            #check the impact and the pval
            if p_val > p_cutoff:
                #print(str(wordID) + ': ' + str(p_val))
                if impact < 0:
                    neg_impact_words.append(wordID)
                    neg_sig_total += (p_val - p_cutoff)
                elif impact > 0:
                    pos_impact_words.append(wordID)
                    pos_sig_total += (p_val - p_cutoff)
        new_topic_num = 0
        if len(pos_impact_words) > (0.1*(len(neg_impact_words))):
            pos_ws_topic_matrix_row = np.zeros((len(word_impact_dict)))
            for wordID in pos_impact_words:
                cur_word_sig = (word_impact_dict[wordID][1] - p_cutoff)
                pos_ws_topic_matrix_row[wordID] = cur_word_sig #calc_prob_word_new_topic(cur_word_sig,pos_sig_total,p_cutoff)
            #word_sig_by_topic_matrix.append(pos_ws_topic_matrix_row)
    
        if len(neg_impact_words) > (0.1*(len(pos_impact_words))):
            neg_ws_topic_matrix_row = np.zeros((len(word_impact_dict)))
            for wordID in neg_impact_words:
                cur_word_sig = (word_impact_dict[wordID][1] - p_cutoff)
                neg_ws_topic_matrix_row[wordID] =  cur_word_sig #calc_prob_word_new_topic(cur_word_sig,neg_sig_total,p_cutoff)
            #word_sig_by_topic_matrix.append(neg_ws_topic_matrix_row)
        
        if np.sum(pos_ws_topic_matrix_row) > 0 and np.sum(neg_ws_topic_matrix_row) > 0:
            new_topic_cnt += 2
            new_pos_row = calc_new_word_prob_for_topic(pos_ws_topic_matrix_row,pos_impact_words)
            new_neg_row = calc_new_word_prob_for_topic(neg_ws_topic_matrix_row,neg_impact_words)
            prior_matrix = np.vstack((prior_matrix,new_pos_row))
            prior_matrix = np.vstack((prior_matrix,new_neg_row))
        elif np.sum(pos_ws_topic_matrix_row) > 0 and np.sum(neg_ws_topic_matrix_row) <= 0:
            new_pos_row = calc_new_word_prob_for_topic(pos_ws_topic_matrix_row,pos_impact_words)
            prior_matrix[topicID] = new_pos_row
        elif np.sum(neg_ws_topic_matrix_row) > 0 and np.sum(pos_ws_topic_matrix_row) <= 0:
            new_neg_row = calc_new_word_prob_for_topic(neg_ws_topic_matrix_row,neg_impact_words)
            prior_matrix[topicID] = new_neg_row

        topic_purities[tp_cnt] = calculate_topic_purity(len(pos_impact_words),len(neg_impact_words))

        tp_cnt += 1

    avg_topic_purity = np.sum(topic_purities)/len(causal_topics)
    #word_sig_total_across_topics_matrix = np.sum(word_sig_by_topic_matrix, axis=0)
    
    #prior_matrix = np.zeros((len(term_topic_matrix)+(len(word_sig_by_topic_matrix)-len(causal_topics)),len(word_impact_dict)))
    #for wordID in range(0,len(word_impact_dict)):
    #    cur_word_sig = word_impact_dict[wordID][1]
    #    if word_sig_total_across_topics_matrix[wordID] > 0:
    #        prior_matrix[wordID] = (cur_word_sig - p_cutoff)/word_sig_total_across_topics_matrix[wordID]

    return prior_matrix,avg_topic_purity,new_topic_cnt



def generate_ITMTF_stats(iteration,tn,avg_causality_confidence,avg_topic_purity):

    try:
        ITMTF_stats_df = pd.read_csv('itmtf_stats.csv')
    except:
        ITMTF_stats_df = pd.DataFrame(columns=['Iteration','tn','Average_Causality_Confidence','Average_Purity'])
    
    new_stats_row = pd.DataFrame(data={'Iteration':[iteration],'tn':[tn],'Average_Causality_Confidence':[avg_causality_confidence],'Average_Purity':[avg_topic_purity]})
    ITMTF_stats_df = ITMTF_stats_df.append(new_stats_row,ignore_index=True)
    ITMTF_stats_df.to_csv('itmtf_stats.csv',index=False)

def list_causal_topics(lda_model,causal_topics,iteration,tn):
    try:
        topic_df = pd.read_csv('causal_topic_words.csv')
    except:
        topic_df = pd.DataFrame(columns=['Iteration','tn','TopicID','Top_Five_Words'])
    
    top_word_list = []
    for topicID in causal_topics:
        top = lda_model.show_topic(topicID,topn=5)
        if top:
            top_words = [w[0] for w in top]
            top_word_str = ' '.join(top_words)
            top_word_list.append(top_word_str)
        else:
            top_word_str = ''
            top_word_list.append(top_word_str)

    iteration_list = [iteration for x in range(len(causal_topics))]
    tn_list = [tn for x in range(len(causal_topics))]

    new_topic_row = pd.DataFrame(data={'Iteration':iteration_list,'tn':tn_list,'TopicID':causal_topics,'Top_Five_Words':top_word_list})
    topic_df = topic_df.append(new_topic_row,ignore_index=True)
    topic_df.to_csv('causal_topic_words.csv',index=False)

def ITMTF(corpus,dictionary,starting_num_topics,num_topics,word_impact_dict,text_time_series,other_time_series,ts_tsID_map,p_cutoff,topic_word_prob_cutoff,iterations,current_iteration = 0,prior_probs = None):
    
    #build LDA model
    lda_model = LdaModel(corpus=corpus,num_topics=num_topics,id2word=dictionary,eta=prior_probs)

    #build the topic stream matrix
    topic_stream_matrix = build_topic_stream(lda_model,corpus,text_time_series,ts_tsID_map)

    #build a dataframe with
    #cols: date, non text TS data value, n-cols: one for each topicID from the LDAModel with values = sum of topic probability for that topic across docs for that day
    #184 rows, one for each day
    topic_causality_df = pd.DataFrame(other_time_series,columns = ['NormalizedPrice','Date'])
    topic_causality_df = topic_causality_df[['Date','NormalizedPrice']]
    for topicID in range(0,num_topics):
        topic_causality_df[topicID] = topic_stream_matrix[topicID]

    #iterate through n-cols, perfrom Pearson test against non text TS column and return topicIDs with p-value greater than some threshold
    #also return the avg p value for significant topics
    causal_topics,avg_causality_confidence = find_causal_topics(num_topics,topic_causality_df,'NormalizedPrice',p_cutoff)

    #build the prior matrix
    prior_matrix,avg_topic_purity,new_topic_cnt = build_prior_matrix(word_impact_dict,causal_topics,lda_model.get_topics(),p_cutoff,topic_word_prob_cutoff)
    if current_iteration > 0:
        generate_ITMTF_stats(current_iteration,starting_num_topics,avg_causality_confidence,avg_topic_purity)
        list_causal_topics(lda_model,causal_topics,current_iteration,starting_num_topics)
    if current_iteration < iterations:
        current_iteration += 1
        ITMTF(corpus,dictionary,starting_num_topics,new_topic_cnt,word_impact_dict,text_time_series,other_time_series,ts_tsID_map,p_cutoff,topic_word_prob_cutoff,iterations,current_iteration=current_iteration,prior_probs=prior_matrix)
    
    return causal_topics


#if __name__ == '__main__':

reload_data = False
#load timeseries datasets
president_norm_stock_ts,gore_bush_nyt_ts = load_data(rebuild=reload_data)

#build a list of raw document strings from the nyt ts dataset
gore_bush_unsplit_doc_list = [t[0] for t in gore_bush_nyt_ts]

#represent the collection as a list of docs represented as a list of tokens while removing stop words [[word1,word2],[word1,word3]]
stop_words = get_stop_words()
#stop_words.extend(['gore','mr','bush','gores','bushs'])
#print('Number of stop_words: ' + str(len(stop_words)))

build_new_corpus = False

gore_bush_doc_list = clean_collection(gore_bush_unsplit_doc_list,stop_words,rebuild=build_new_corpus)

#build the gensim Dictionary and corpus (ID representation for words and docs + word counts)
gore_bush_gensim_dictionary,gore_bush_gensim_corpus = build_gensim_corpus(gore_bush_doc_list,rebuild=build_new_corpus)

ts_tsID_map = {}
for x in range(0,len(president_norm_stock_ts)):
    ts_tsID_map[president_norm_stock_ts[x][1]] = x


if build_new_corpus == True:
    #initiate word-level causality analysis
    #build the word count stream matrix 
    wc_stream_matrix = build_word_count_stream(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,gore_bush_nyt_ts,ts_tsID_map)
    num_words_in_collection = len(wc_stream_matrix)
    
    #build a dataframe with
    #cols: date, non text TS data value, n-cols: one for each wordID from the corpus vocabulary with values = sum of word counts for that word across docs for that day
    #184 rows, one for each day
    causality_df = pd.DataFrame(president_norm_stock_ts,columns = ['NormalizedPrice','Date'])
    word_causality_df = causality_df[['Date','NormalizedPrice']]
    for wordID in range(0,num_words_in_collection):
        word_causality_df[wordID] = wc_stream_matrix[wordID]
    
    #iterate through n-cols, perfrom Pearson test against non text TS column and return a dict of {wordID:(impact value,p-value)} if the pvalue for the word is greater than some threshold
    word_impact_dict = build_word_impact_dict(num_words_in_collection,word_causality_df,'NormalizedPrice')
    pickle.dump(word_impact_dict, open('word_impact_dict.pkl', 'wb'))
    print('word_impact_dict built...')
else:
    try:
        word_impact_dict = pickle.load(open('word_impact_dict.pkl', 'rb'))
        print('word_impact_dict loaded...')
    except:
        #initiate word-level causality analysis
        #build the word count stream matrix 
        wc_stream_matrix = build_word_count_stream(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,gore_bush_nyt_ts,ts_tsID_map)
        num_words_in_collection = len(wc_stream_matrix)
        
        #build a dataframe with
        #cols: date, non text TS data value, n-cols: one for each wordID from the corpus vocabulary with values = sum of word counts for that word across docs for that day
        #184 rows, one for each day
        causality_df = pd.DataFrame(president_norm_stock_ts,columns = ['NormalizedPrice','Date'])
        word_causality_df = causality_df[['Date','NormalizedPrice']]
        for wordID in range(0,num_words_in_collection):
            word_causality_df[wordID] = wc_stream_matrix[wordID]
        
        #iterate through n-cols, perfrom Pearson test against non text TS column and return a dict of {wordID:(impact value,p-value)} if the pvalue for the word is greater than some threshold
        word_impact_dict = build_word_impact_dict(num_words_in_collection,word_causality_df,'NormalizedPrice')
        pickle.dump(word_impact_dict, open('word_impact_dict.pkl', 'wb'))
        print('word_impact_dict built...')

#begin ITMTF
min_significance_value = 0.8
min_topic_prob = 0.001
iterations = 5

number_of_topics = 10
causal_topics = ITMTF(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,number_of_topics,number_of_topics,word_impact_dict,gore_bush_nyt_ts,president_norm_stock_ts,ts_tsID_map,min_significance_value,min_topic_prob,iterations)

#begin ITMTF
number_of_topics = 20
causal_topics = ITMTF(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,number_of_topics,number_of_topics,word_impact_dict,gore_bush_nyt_ts,president_norm_stock_ts,ts_tsID_map,min_significance_value,min_topic_prob,iterations)

#begin ITMTF
number_of_topics = 30
causal_topics = ITMTF(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,number_of_topics,number_of_topics,word_impact_dict,gore_bush_nyt_ts,president_norm_stock_ts,ts_tsID_map,min_significance_value,min_topic_prob,iterations)

#begin ITMTF
number_of_topics = 40
causal_topics = ITMTF(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,number_of_topics,number_of_topics,word_impact_dict,gore_bush_nyt_ts,president_norm_stock_ts,ts_tsID_map,min_significance_value,min_topic_prob,iterations)

#begin ITMTF
number_of_topics = 100
causal_topics = ITMTF(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,number_of_topics,number_of_topics,word_impact_dict,gore_bush_nyt_ts,president_norm_stock_ts,ts_tsID_map,min_significance_value,min_topic_prob,iterations)