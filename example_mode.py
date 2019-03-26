import praw
import pandas as pd
import numpy as np
import sys
import re
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from praw.models import MoreComments
from flask import Flask, abort, request
from uuid import uuid4

import requests
import requests.auth
import urllib


def model_predict(INPUT):
    
    prediction_input_name = INPUT
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
    new_word_counts = new_word_counts.rename(columns=pd.Series(prediction_input_name))
    edited_input_mat = pd.read_csv('input features pre-regularization for dict_50.csv',index_col=0)
    x_predict = pd.concat([new_word_counts,edited_input_mat],axis=1,sort=True)
    x_predict = x_predict.fillna(0)
    x_predict = x_predict.iloc[:,0]
    x_predict = x_predict[x_predict.index.isin(edited_input_mat.index)]
    x_predict = pd.DataFrame(x_predict).transpose()
    model = load_model('first_NN_with_dict_50.h5')
    prediction = model.predict(x_predict)
    prediction = pd.DataFrame(prediction)
    prediction = prediction.rename(columns=pd.Series(y_train.index.values),index=pd.Series('xetrin'))
    prediction = prediction.transpose()
    prediction.to_csv('xetrin subreddit recommendations.csv',header=True,index=True,mode='w')
 
argList = sys.argv
input_string = argList[1]
model_predict(input_string)
