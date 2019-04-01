#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding


# define constant variables 
max_words = 30
top_words = 150000
embedding_vector_length = 32 


# read both train and test datasets
def read_csv_data():
    df_test = pd.read_csv('test.csv') # read test into csv 
    df_train = pd.read_csv('train.csv') # read train into csv 
    return (df_train, df_test)

df_train,df_test = read_csv_data()


# drop category of train data before training 
def sieveCategories(df_train):
    # sieve categories out 
    df_train_category = df_train['Category']
    #print(df_train_category.head())
    df_temp = df_train.drop(columns = "Category")
    return (df_temp,df_train_category)

df_temp, df_train_category = sieveCategories(df_train)


# get length of training data. Needed for deconcatenation 
def getLenCsv(df_train, df_test):
    return (len(df_train.index), len(df_test.index))

len_train, len_test = getLenCsv(df_train, df_test)

# concatenate train and test set for tokenization 
def concatenate_and_tokenize(df_temp,df_test):
    result = pd.concat([df_temp,df_test]) # concat the two dataframes 
    # concatenate the two titles together 
    tokenizer = Tokenizer()
    titles = result['title']
    tokenizer.fit_on_texts(titles)
    result['title'] = tokenizer.texts_to_sequences(titles)
    #print(len(result.index))
    #print(result['title'][0])
    #print(result.tail())
    return result 

result = concatenate_and_tokenize(df_temp, df_test)

# Deconcatenate train and test set for predictions and training
def deconcatenate_train_and_test(result):
    df_train_deconcat = result.iloc[:len_train, :]
    #print(df_train_deconcat.head())
    df_test_deconcat = result.iloc[len_train:, :]
    #print(df_test_deconcat.head())
    return (df_train_deconcat, df_test_deconcat)

df_train_deconcat, df_test_deconcat = deconcatenate_train_and_test(result)





# add category from train csv back to train data 
def add_back_categories(df_train_deconcat, df_train_category):
    df_train_deconcat['Category'] = df_train_category
    #print(df_train_deconcat.head())
    #print(df_test_deconcat.head())
    
add_back_categories(df_train_deconcat, df_train_category)


# function that makes model 
def make_model(offset,num_output,num_epochs,image_path):
    mobile_data = df_train_deconcat.loc[df_train_deconcat['image_path'].str.contains(image_path)] # train for mobile 
    #print(mobile_data.size)
    #print(mobile_data.head())
    y = mobile_data['Category'] # get mobile data category 
    y = np.array(y.values)   # make into numpy array 
    y = y - offset           # find offset 
    temp = np.zeros((y.size, y.max() + 1))
    temp[np.arange(y.size), y] = 1 
    y_output = temp 
    #print(y_output)
    X_train = mobile_data['title'].values
    X_train = sequence.pad_sequences(X_train,maxlen = max_words)
    #print(X_train)
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length = max_words))
    model.add(LSTM(100))
    model.add(Dense(num_output, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])
    model.summary()
    model.fit(X_train,y_output, epochs = num_epochs, batch_size = 64)
    return model
   
 
# make_model(offset, num_output, epochs)
# can change num of epochs i was just testing 
def make_models():
    model_dict = {}
    model_fashion = make_model(17,14,25,"fashion_image")
    model_beauty = make_model(0,17,25,"beauty_image")
    model_mobile = make_model(31,27,25,"mobile_image")
    model_dict = { "mobile_image": model_mobile, "model_beauty": model_beauty, "model_fashion": model_fashion }
    return model_dict

final_dict = make_models()


def make_predict_and_save(model_dict,df_test_deconcat):
    #model_dict = { "mobile_image": model_mobile, "model_beauty": None, "model_fashion": None }  
    # make predictions for each of the model 
    predictions = np.empty((0,2), int)
    #print(predictions)
    offset_dict = {"mobile_image": 31, "fashion_image": 17, "beauty_image": 0}
    X_test = df_test_deconcat['title'].values
    X_test = sequence.pad_sequences(X_test,maxlen = max_words)
    #print(X_test)
    for (i,row) in df_test_deconcat.iterrows():
        #print(row)
        image = df_test_deconcat['image_path'][i].split('/')[0]  # get the image path before "/"" 
        #print(image)
        convertor = {'beauty_image':'model_beauty','fashion_image':'model_fashion','mobile_image':'mobile_image'}
        model_to_train = model_dict[convertor[image]] # get the corresponding model 
        numpy_array = model_to_train.predict( np.array ( [X_test[i], ]) ) # returns a numpy array of prediction values 
        index = np.argmax(numpy_array) # take the maximum predicted values 
        category = index + offset_dict[image]  # get the offset corresponding to the different image sets 
        predictions = np.append(predictions, np.array([[row['itemid'],category]]), axis = 0) # make a prediction and append that to the numpy array
        #print(predictions)
    #print(predictions)
    pd.DataFrame(predictions, columns = ['itemid', 'predictions']).to_csv('predictions.csv') # make into csv_file after everything is done 

make_predict_and_save(final_dict, df_test_deconcat)

