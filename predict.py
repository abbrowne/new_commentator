import praw
import pandas as pd
import numpy as np
import sys
import re
import tensorflow as tf
import keras
import random
from keras.models import Sequential, load_model
from keras.layers import Dense
from praw.models import MoreComments
from flask import Flask, abort, request
from uuid import uuid4

import requests
import requests.auth
import urllib


def authenticate():
    print('Authenticating...\n')
    reddit = praw.Reddit('commentator',user_agent='testscript by /u/xetrin')
    print('Authenticated as {}\n'.format(reddit.user.me()))
    return reddit

def model_predict(input_string, reddit):
    output_filename = input_string + ' subreddit recommendations.html'
    state = str(random.randint(0, 65000))
    scopes = ['identity']
    prediction_input_name = input_string
    author_comment_mat = pd.DataFrame()
    for comment in reddit.redditor(name=prediction_input_name).comments.new(limit=None):
        author_comment_mat = author_comment_mat.append(pd.Series([comment.body,comment.author,comment.subreddit,comment.id]), ignore_index=True)
    
    author_comment_mat = author_comment_mat.rename(columns=pd.Series(['comment','author','subreddit','comment_id']))
    author_subreddits = pd.Series(author_comment_mat['subreddit'].unique())
    author_subreddit_mat = pd.DataFrame(np.ones(len(author_subreddits)))
    author_subreddit_mat = author_subreddit_mat.rename(index=author_subreddits,columns=pd.Series(prediction_input_name))
    y_train = pd.read_csv('Subreddits for the top 1000 redditors.csv',index_col=0)
    y_predict = pd.concat([author_subreddit_mat,y_train],axis=1,sort=True)
    y_predict = y_predict.fillna(0)
    y_predict = y_predict.iloc[:,0]
    y_predict = y_predict[y_predict.index.isin(y_train.index)]
    y_predict = pd.DataFrame(y_predict)
    author_comment_mat = author_comment_mat.set_index(author_comment_mat['comment_id'])
    author_comment_lines = pd.DataFrame(author_comment_mat.iloc[:,0].copy())
    author_comment_lines = author_comment_lines.set_index(author_comment_mat.index.values)
    author_comment_lines = author_comment_lines.dropna()
    author_comment_lines['comment'] = author_comment_lines['comment'].map(lambda x: re.sub(r'\W+', ' ', x)).str.replace('  ',' ').str.strip().str.lower()
    author_comment_lines = author_comment_lines.drop_duplicates()
    ##Split comments by spaces into lists of unique words
    
    author_comment_words = pd.Series(author_comment_lines['comment'].str.split(' '))
    result = pd.Series(list(set(x for l in author_comment_words for x in l)))
    result = result[result.values != '']
    result = result[result.values != ' ']
    words = pd.Series(result.values.copy())
    id_list = pd.Series(author_comment_words.index.values)
    ##Generate a comment by word matrix with 1(TRUE) or 0(FALSE) values for the presence of each word in each comment
    
    word_comment_mat = pd.DataFrame()
    full_word_comment_mat = pd.DataFrame()
    temp_comment = 0
    for comment in np.arange(0,len(author_comment_words)):
        temp_words = pd.DataFrame(result.isin(pd.Series(author_comment_words.iloc[comment])))
        temp_words = temp_words.rename(columns=pd.Series(prediction_input_name))
        word_comment_mat = pd.concat([word_comment_mat,temp_words],axis=1)
        if(((comment+1)%1000==0) | ((comment+1) == len(author_comment_words))):
            word_comment_mat = word_comment_mat.rename(index=words)
            full_word_comment_mat = pd.concat([full_word_comment_mat,word_comment_mat],axis=1)
            temp_comment = comment.copy()
            word_comment_mat = pd.DataFrame()
        
    
    print(len(author_comment_words))
    full_word_comment_mat = full_word_comment_mat.astype('int64')
    new_word_counts = pd.DataFrame(full_word_comment_mat.sum(axis=1))
    new_word_counts.iloc[:,0] = 1
    new_word_counts = new_word_counts.rename(columns=pd.Series(prediction_input_name))
    new_word_counts = new_word_counts.sort_index()
    new_word_counts = new_word_counts
    edited_input_mat = pd.read_csv('X train normalized for dict_50.csv',index_col=0)
    edited_input_mat = edited_input_mat.sort_index()
    if (new_word_counts.columns.isin(edited_input_mat.columns)):
        edited_input_mat = edited_input_mat.drop([input_string],axis=1)
    
    x_predict = pd.concat([new_word_counts,edited_input_mat],axis=1,sort=True)
    x_predict = x_predict.fillna(0).transpose()
    x_predict = x_predict.loc[input_string]
    x_predict = 2 * (x_predict - 0.5)
    x_predict = x_predict[x_predict.index.isin(edited_input_mat.index)]
    x_predict = x_predict[~x_predict.index.duplicated()].sort_index()
    x_predict = pd.DataFrame(x_predict).transpose()
    model = load_model('new_dict_binary_NN.h5')
    prediction = model.predict(x_predict)
    prediction = pd.DataFrame(prediction)
    prediction = prediction.rename(columns=pd.Series(y_train.index.values),index=pd.Series(input_string))
    prediction = prediction.transpose()
    prediction = prediction[prediction.values > 0.2].sort_values(input_string,ascending=False)
    keras.backend.clear_session()
    x_predict_mod = (x_predict + 1) / 2
    x_predict_mod = x_predict_mod.transpose()
    x_predict_mod = x_predict_mod.drop(x_predict_mod[x_predict_mod.loc[:,input_string] < 1].index)
    network_input = pd.Series(prediction.index.values)
    network_input = plot_network(network_input,x_predict_mod)
    network_input[0].to_csv('edge_mat.csv',header=True,index=True,mode='w')
    network_input[1].to_csv('node_mat.csv',header=True,index=True,mode='w')
    return prediction.to_html(),network_input[0].to_html(),network_input[1].to_html(),'simple_path.png'

def main(user_input):
    reddit = authenticate()
    return model_predict(user_input, reddit)


def plot_network(predicted_subR,query_input):
    x_train = pd.read_csv('X train normalized for dict_50.csv',index_col=0)
    x_train = x_train.sort_index(axis=0)
    x_train = x_train.sort_index(axis=1)
    y_train = pd.read_csv('Subreddits for the top 1000 redditors.csv',index_col=0)
    y_train = y_train.sort_index(axis=0)
    y_train = y_train.sort_index(axis=1)
    query_words = pd.Series(query_input.index.values)
    subR_list = pd.Series(predicted_subR).sort_values()
    subR_words = pd.DataFrame(np.zeros((len(subR_list),x_train.shape[0])),index=subR_list.values,columns=x_train.index.values)
    edge_mat = pd.DataFrame()
    node_mat = pd.DataFrame(columns=subR_list.values)
    top_10 = pd.DataFrame(columns=subR_list.values)
    x_mod = (x_train + 1) / 2
    node_sizes = y_train.sum(axis=1)
    word_popularity = x_mod.sum(axis=1)
    for i in np.arange(0,len(subR_list)):
        print('i equals ' + str(i))
        tmp_subR = y_train.loc[subR_list[i],:] == 1
        tmp_subR = y_train.iloc[:,tmp_subR.values]
        for j in np.arange(0,tmp_subR.shape[1]):
            subR_words.loc[subR_list[i],:] = subR_words.loc[subR_list[i],:].add(x_mod.iloc[:,j],fill_value=0)
        
        subR_words = subR_words.loc[:,query_words.values]
        tmp_10 = subR_words.loc[subR_list[i],:].sort_values(ascending=False)
        top_10.loc[:,subR_list[i]] = pd.Series(tmp_10.index.values[0:10])
        tmp_nodes = pd.Series([subR_list[i],node_sizes.loc[subR_list[i]],top_10.loc[:,subR_list[i]]])
        node_mat.loc[:,subR_list[i]] = tmp_nodes
        
        for k in np.arange((i+1),len(subR_list)):
            tmp_weight = 0
            tmp_subR2 = tmp_subR.loc[subR_list[k],:] == 1
            tmp_subR2 = tmp_subR.iloc[:,tmp_subR2.values]
            tmp_weight = tmp_subR2.shape[1]
            if (tmp_weight >= 1):
                tmp_edge = pd.Series([subR_list[i],subR_list[k],tmp_weight])
                edge_mat = pd.concat([edge_mat,tmp_edge],axis=1)
            
        
    
    return edge_mat, node_mat


