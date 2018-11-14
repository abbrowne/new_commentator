#activate tensorflow-gpu

import praw
import pandas as pd
import datetime as dt
import numpy as np
import os
import sys
import re
import csv
from praw.models import MoreComments
import predict
from stemming.porter2 import stem
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model

## Parameters to set

redditors_for_training = 1000

##Login info

reddit = praw.Reddit('commentator',user_agent='testscript by /u/xetrin')

#####Code for generating dict from top submission comments

top_redditors_mat = pd.DataFrame()
for submission in reddit.subreddit('all').hot(limit=redditors_for_training):
    top_redditors_mat = top_redditors_mat.append(pd.Series([submission, submission.title, submission.author, submission.subreddit, submission.subreddit_id]),ignore_index=True)

top_redditors_mat = top_redditors_mat.rename(columns=pd.Series(['submission','title','author','subreddit','subreddit_id']))
top_sub_filename = "Top " + str(redditors_for_training) + " submissions.csv"
top_redditors_mat.to_csv('Top 1000 submissions.csv',header=True,index=True,mode='w')
submission_ids = top_redditors_mat['submission']
#########Code for generation of training set (X and Y)

a_ids = top_redditors_mat['author']
all_word_counts = pd.DataFrame()
y_train = pd.DataFrame()
a_id = 0
while a_id < len(a_ids):
    a_cmnts = pd.DataFrame()
    while(a_ids.iloc[a_id] == None):
        a_id += 1
    while a_ids.iloc[a_id].name in all_word_counts.columns.values:
        a_id += 1
    try:
        for comment in reddit.redditor(name=a_ids.iloc[a_id].name).comments.new(limit=None):
            temp_cmnt = pd.Series([comment.body,comment.author,comment.subreddit,comment.id])
            a_cmnts = a_cmnts.append(temp_cmnt, ignore_index=True)
    except:
        a_id += 1
    while(len(a_cmnts) == 0):
        a_id += 1
        for comment in reddit.redditor(name=a_ids.iloc[a_id].name).comments.new(limit=None):
            a_cmnts = a_cmnts.append(pd.Series([comment.body,comment.author,comment.subreddit,comment.id]), ignore_index=True)
    a_cmnts = a_cmnts.rename(columns=pd.Series(['comment','author','subreddit','comment_id']))
    a_subrs = pd.Series(a_cmnts['subreddit'].unique())
    a_subr_mat = pd.DataFrame(np.ones(len(a_subrs)))
    a_subr_mat = a_subr_mat.rename(index=a_subrs,columns=pd.Series([a_ids.iloc[a_id].name]))
    y_train = pd.concat([y_train,a_subr_mat],axis=1,sort=True)
    y_train = y_train.fillna(0)
    a_cmnts = a_cmnts.set_index(a_cmnts['comment_id'])
    a_lines = pd.DataFrame(a_cmnts.iloc[:,0].copy())
    a_lines = a_lines.set_index(a_cmnts.index.values)
    a_lines = a_lines.dropna()
    a_lines['comment'] = a_lines['comment'].map(lambda x: re.sub(r'\W+', ' ', x)).str.replace('  ',' ').str.strip().str.lower()
    a_lines = a_lines.drop_duplicates()
    ##Split comments by spaces into lists of unique words
    a_comment_words = pd.Series(a_lines['comment'].str.split(' '))
    result = pd.Series(list(set(x for l in a_comment_words for x in l)))
    result = result[result.values != '']
    result = result[result.values != ' ']
    words = pd.Series(result.values.copy())
    id_list = pd.Series(a_comment_words.index.values)
    ##Generate a comment by word matrix with 1(TRUE) or 0(FALSE) values for the presence of each word in each comment
    word_comment_mat = pd.DataFrame()
    full_word_comment_mat = pd.DataFrame()
    temp_comment = 0
    for comment in np.arange(0,len(a_comment_words)):
        temp_words = pd.DataFrame(result.isin(pd.Series(a_comment_words.iloc[comment])))
        temp_words = temp_words.rename(columns=pd.Series(a_ids.iloc[a_id].name))
        word_comment_mat = pd.concat([word_comment_mat,temp_words],axis=1)
        if(((comment+1)%1000==0) | ((comment+1) == len(a_comment_words))):
            word_comment_mat = word_comment_mat.rename(index=words)
            full_word_comment_mat = pd.concat([full_word_comment_mat,word_comment_mat],axis=1)
            temp_comment = comment.copy()
            word_comment_mat = pd.DataFrame()
    full_word_comment_mat = full_word_comment_mat.astype('int64')
    new_word_counts = pd.DataFrame(full_word_comment_mat.sum(axis=1))
    new_word_counts.iloc[:,0] = 1
    new_word_counts = new_word_counts.rename(columns=pd.Series([a_ids.iloc[a_id]]))
    all_word_counts = pd.concat([all_word_counts,new_word_counts],axis=1,sort=True)
    all_word_counts = all_word_counts.fillna(0)
    print(a_ids.iloc[a_id])
    print(a_id)
    a_id += 1

y_train.to_csv('Subreddits for the top 1000 redditors.csv',header=True,index=True,mode='w')
all_word_counts.to_csv('Comment word frequency for comments from top 1000 redditors.csv',header=True,index=True,mode='w')
test = pd.DataFrame(all_word_counts.sum(axis=1))
test2 = pd.read_csv('Top 1000 submission comment word frequency backup.csv',index_col=0)
test3 = test2.groupby(test2.index).max()
merged_word_counts = pd.concat([test,test3],axis=1,sort=True)
merged_word_counts.fillna(0)
merged_word_counts.to_csv('Merged word frequencies backup.csv',header=True,index=True,mode='w')
###Dictionary generation

merged_word_counts = pd.read_csv('Merged word frequencies backup.csv',index_col=0)
dict_50 = merged_word_counts[(merged_word_counts.iloc[:,0] >= 10) & (merged_word_counts.iloc[:,1] <= 50)]
dict_50.to_csv('Word dictionary with frequency of 10 or greater.csv',header=True,index=True,mode='w')
##Filter and regularize

all_word_counts = pd.read_csv('Comment word frequency for comments from top 1000 redditors.csv',index_col=0)
prereg_edited_input_mat = all_word_counts[all_word_counts.index.isin(dict_50.index)]
prereg_edited_input_mat.to_csv("Prereg input features for dict.csv",header=True,index=True,mode='w')

############Neural network training

edited_input_mat = 2 * (prereg_edited_input_mat - 0.5)
edited_input_mat = edited_input_mat.groupby(edited_input_mat.index).max()
x_train = edited_input_mat.copy()
x_train.to_csv('X train normalized for dict_50.csv',header=True,index=True,mode='w')

x_train = pd.read_csv('X train normalized for dict_50.csv',index_col=0)
y_train = pd.read_csv('Subreddits for the top 1000 redditors.csv',index_col=0)
model = Sequential()
model.add(Dense(1000, input_dim=len(x_train), activation='relu'))
model.add(Dense(len(y_train), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train = x_train.transpose()
y_train = y_train.transpose()
model.fit(x_train, y_train, epochs=100, batch_size=1)
model.save("new_dict_binary_NN.h5")
keras.backend.clear_session()