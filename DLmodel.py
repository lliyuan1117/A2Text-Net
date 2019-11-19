#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Deep Learning necessities
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional, Dropout, Conv1D, MaxPool1D
from keras.layers import GlobalMaxPool1D, GRU, Input, Concatenate, Conv2D
from keras import optimizers, Model


import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
from string import punctuation
from collections import Counter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Deep Learning necessities
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional, Dropout, Conv1D, MaxPool1D
from keras.layers import GlobalMaxPool1D, GRU, Input, Concatenate, Conv2D
from keras import optimizers, Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
from string import punctuation
from collections import Counter
from scipy import interp

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential


import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn



from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

# from sklearn.grid_search import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[2]:


def prep_data(text, tok):
    seq = tok.texts_to_sequences([text])
    data = pad_sequences(seq, MAX_SEQ_LENGTH)
    return data

def plot(history):
    hist = history.history
    train_loss, train_acc = hist['loss'], hist['acc']
    val_loss, val_acc = hist['val_loss'], hist['val_acc']
    epochs = range(1, len(train_acc)+1)
    
    plt.plot(epochs, train_acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
def TPR_FPR(model, X_test, y_test):
    y_score = model.predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     roc_auc["micro"] = auc(np.arrary(y_test), np.arrary(y_score))
    
    return fpr["micro"], tpr["micro"], roc_auc["micro"]

def precision_recall(model, X_test, y_test):
    y_score = model.predict_classes(X_test)
    precision, recall,f1, _ = precision_recall_fscore_support(y_test, y_score,average='weighted',labels=np.unique(y_score))
#     average_precision = average_precision_score(y_test.ravel(), y_score.ravel())
    return precision, recall,f1, _


# In[5]:


df = pd.read_csv('ghosh data.csv', encoding='latin-1')


# In[6]:


df = df[['text', 'sarcasm']]

EMB_DIMMAX_WORDS = 20000

tok = Tokenizer(num_words = 10000) # keeping 10000 now for first iteration
tok.fit_on_texts(df.text)
seqs = tok.texts_to_sequences(df.text)

# Find length of sentence 
df['length'] = df['text'].apply(lambda x: len(x.split(' ')))


# In[7]:


max(df['length'])


MAX_SEQ_LENGTH = 327
data = pad_sequences(seqs, MAX_SEQ_LENGTH)
labels = np.asarray(df.sarcasm)
data.shape


def logsitic():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS,output_dim=EMB_DIM, input_length=MAX_SEQ_LENGTH))
    model.add(Flatten())
    model.add(Dense(1,  # output dim is 2, one score per each class
                activation='sigmoid',
                kernel_regularizer=L1L2(l1=0.0, l2=0.5),
                input_dim=3)) 
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


# In[14]:


from keras.regularizers import L1L2
Dense_FPR = []

Dense_TPR = []

Dense_ROC = []

Dense_Recall = []

Dense_f1 = []
tprs= []

Dense_Precision=[]
# dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(fcmod,x_val,y_val)
# lst_fpr,lst_tpr,lst_roc_auc = TPR_FPR(lsmod, x_val, y_val)
# conv_fpr,conv_tpr,conv_roc_auc  = TPR_FPR(convmod, x_val, y_val)
# congru_fpr,congru_tpr,congru_roc_auc  = TPR_FPR(convgrumod, x_val, y_val)
# log_fpr,log_tpr,log_auc=TPR_FPR(log,x_val,y_val)


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)


kf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
for train_index, test_index in kf.split(data,labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    
    logmod = logsitic()
    fchist= logmod.fit(X_train, y_train,
         epochs = 10,
         batch_size = 256,
         validation_data = (X_test, y_test))
    
    
#     print(classification_report(y_test,fcmod.predict(X_test).argmax(axis=-1)))
    
    dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(logmod,X_test, y_test)
    precision, recall,f1,_ = precision_recall(logmod,X_test, y_test)
    tprs.append(interp(mean_fpr, dense_fpr, dense_tpr))
    Dense_Recall.append(recall)
    Dense_f1.append(f1)
    Dense_Precision.append(precision)
    Dense_FPR.append(dense_fpr)
    Dense_TPR.append(dense_tpr)
    Dense_ROC.append(dense_roc_auc)


# In[15]:


print("The Five-Cross Validation Average F1 Score is " , Average(Dense_f1))
print("The Five-Cross Validation Average Precesion Score is " , Average(Dense_Precision))
print("The Five-Cross Validation Average Recall Score is " , Average(Dense_Recall))
print("The Five-Cross Validation Average ROC Score is " , Average(Dense_ROC))


# In[16]:


EMB_DIM = 6
MAX_WORDS=10000
def fcmodel():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim= EMB_DIM, input_length=MAX_SEQ_LENGTH))    
    
    # Flatten Layer
    model.add(Flatten())
    
    # FC1
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # print model summary
#     model.summary()
    
    # When using pretrained embeddings
    #model.layers[0].set_weights([embedding_matrix])
    #model.layers[0].trainable = False
              
    # Compile the model
    model.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['acc'])
    return model


# In[17]:


from sklearn.model_selection import StratifiedKFold
Dense_FPR = []

Dense_TPR = []

Dense_ROC = []

Dense_Recall = []

Dense_f1 = []
tprs= []

Dense_Precision=[]
# dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(fcmod,x_val,y_val)
# lst_fpr,lst_tpr,lst_roc_auc = TPR_FPR(lsmod, x_val, y_val)
# conv_fpr,conv_tpr,conv_roc_auc  = TPR_FPR(convmod, x_val, y_val)
# congru_fpr,congru_tpr,congru_roc_auc  = TPR_FPR(convgrumod, x_val, y_val)
# log_fpr,log_tpr,log_auc=TPR_FPR(log,x_val,y_val)


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)


kf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
for train_index, test_index in kf.split(data,labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    fcmod = fcmodel()
    fchist= fcmod.fit(X_train, y_train,
         epochs = 10,
         batch_size = 256,
         validation_data = (X_test, y_test))
    
    
#     print(classification_report(y_test,fcmod.predict(X_test).argmax(axis=-1)))
    
    dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(fcmod,X_test, y_test)
    precision, recall,f1,_ = precision_recall(fcmod,X_test, y_test)
    tprs.append(interp(mean_fpr, dense_fpr, dense_tpr))
    Dense_Recall.append(recall)
    Dense_f1.append(f1)
    Dense_Precision.append(precision)
    Dense_FPR.append(dense_fpr)
    Dense_TPR.append(dense_tpr)
    Dense_ROC.append(dense_roc_auc)


# In[18]:


print("The Five-Cross Validation Average F1 Score is " , Average(Dense_f1))
print("The Five-Cross Validation Average Precesion Score is " , Average(Dense_Precision))
print("The Five-Cross Validation Average Recall Score is " , Average(Dense_Recall))
print("The Five-Cross Validation Average ROC Score is " , Average(Dense_ROC))


# In[19]:


def lstm():
    model = Sequential()
    
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMB_DIM, input_length=MAX_SEQ_LENGTH))
    
    model.add(Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(32, recurrent_dropout=0.1, dropout=0.1)))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer = 'rmsprop',
                  loss = 'binary_crossentropy',
                  metrics = ['acc'])
    
    return model


# In[21]:


Dense_FPR = []

Dense_TPR = []

Dense_ROC = []

Dense_Recall = []

Dense_f1 = []
tprs= []

Dense_Precision=[]
# dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(fcmod,x_val,y_val)
# lst_fpr,lst_tpr,lst_roc_auc = TPR_FPR(lsmod, x_val, y_val)
# conv_fpr,conv_tpr,conv_roc_auc  = TPR_FPR(convmod, x_val, y_val)
# congru_fpr,congru_tpr,congru_roc_auc  = TPR_FPR(convgrumod, x_val, y_val)
# log_fpr,log_tpr,log_auc=TPR_FPR(log,x_val,y_val)


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)


kf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
for train_index, test_index in kf.split(data,labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    lsmod = lstm()
    fchist= lsmod.fit(X_train, y_train,
         epochs = 10,
         batch_size = 256,
         validation_data = (X_test, y_test))
    
    dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(lsmod,X_test, y_test)
    precision, recall,f1,_ = precision_recall(lsmod,X_test, y_test)
    tprs.append(interp(mean_fpr, dense_fpr, dense_tpr))
    Dense_Recall.append(recall)
    Dense_f1.append(f1)
    Dense_Precision.append(precision)
    Dense_FPR.append(dense_fpr)
    Dense_TPR.append(dense_tpr)
    Dense_ROC.append(dense_roc_auc)


# In[22]:


print("The Five-Cross Validation Average F1 Score is " , Average(Dense_f1))
print("The Five-Cross Validation Average Precesion Score is " , Average(Dense_Precision))
print("The Five-Cross Validation Average Recall Score is " , Average(Dense_Recall))
print("The Five-Cross Validation Average ROC Score is " , Average(Dense_ROC))


# In[23]:


df = pd.read_csv('ghosh data.csv', encoding='latin-1')
df.head()


# In[24]:


df["S_Quotations"] = df.text.str.count("\'")
df["Quotations"] = df.text.str.count('"')
df["Comma"] = df.text.str.count(',')
df["Dash"] = df.text.str.count('-')
df["Colon"] = df.text.str.count(':')
df["Period"] = df.text.str.count('\.')
df["Dollor"] = df.text.str.count('$')
df["Question"] = df.text.str.count('\?')
df["jinghao"]=df.text.str.count('#')
df["shangyinhao"]=df.text.str.count('""')
df["xiahuaxian"]=df.text.str.count('_')
df["at"]=df.text.str.count('@')
df["gantan"]=df.text.str.count('!')
df["star"]=df.text.str.count('\*')
df["and"]=df.text.str.count('&')
df["youkuahao"]=df.text.str.count('\)')
df["zuokuahao"]=df.text.str.count('\(')
df["xie"]=df.text.str.count('/')
df["fenhao"]=df.text.str.count(';')
df["percent"]=df.text.str.count('%')
df["equal"]=df.text.str.count('=')
df["plus"]=df.text.str.count('\+')
df["bolang"]=df.text.str.count('~')
df["guaihao"]=df.text.str.count('^')
df["youfangkkuahao"]=df.text.str.count(']')
df["zuofangkuahao"]=df.text.str.count('\[')
df["shuxian"]=df.text.str.count('\|')


# In[25]:


punc = df[["S_Quotations","Quotations","Comma","Dash","Colon","Period","Dollor","Question","jinghao","shangyinhao","xiahuaxian","at","gantan","star","and","youkuahao","zuokuahao","xie","fenhao","percent","equal","plus","bolang","guaihao","youfangkkuahao","zuofangkuahao","shuxian"]].values


# In[26]:


sar_news = []
for rows in range(0, df.shape[0]):
    head_txt = df.text[rows]
    head_txt = head_txt.split(" ")
    sar_news.append(head_txt)


# In[27]:


from nltk.tag import pos_tag, map_tag
pos_sarc = []
for i in sar_news:
    pos_sarc.append(pos_tag([j for j in i if j]))

print(pos_sarc)


# In[28]:


new_pos = []
all_in = []
for i in pos_sarc:
    tep =[]
    for j in i:
        tep.append(j[1])
        all_in.append(j[1])
    new_pos.append(tep)
print(len(new_pos))


# In[29]:


np.unique(all_in,return_counts = True)


# In[30]:


def count_pos(text, target):
    temp = []
    for i in text:
        temp.append(i.count(target))
    
    return temp


# In[31]:


df["NN"] = count_pos(new_pos, "NN")


# In[32]:


df["CC"]=count_pos(new_pos, "CC")


# In[33]:


df["CD"]=count_pos(new_pos, "CD")


# In[34]:


df["DT"]=count_pos(new_pos, "DT")


# In[35]:


df["EX"]=count_pos(new_pos, "EX")
df["FW"]=count_pos(new_pos, "FW")
df["IN"]=count_pos(new_pos, "IN")
df["JJ"]=count_pos(new_pos, "JJ")
df["JJR"]=count_pos(new_pos, "JJR")
df["JJS"]=count_pos(new_pos, "JJS")
df["MD"]=count_pos(new_pos, "MD")
df["NNP"]=count_pos(new_pos, "NNP")
df["NNPS"]=count_pos(new_pos, "NNPS")
df["NNS"]=count_pos(new_pos, "NNS")
df["PDT"]=count_pos(new_pos, "PDT")
df["POS"]=count_pos(new_pos, "POS")
df["PRP"]=count_pos(new_pos, "PRP")
df["PRP$"]=count_pos(new_pos, "PRP$")
df["RB"]=count_pos(new_pos, "RB")
df["RBR"]=count_pos(new_pos, "RBR")
df["RBS"]=count_pos(new_pos, "RBS")
df["RP"]=count_pos(new_pos, "RP")
df["SYM"]=count_pos(new_pos, "SYM")
df["TO"]=count_pos(new_pos, "TO")
df["UH"]=count_pos(new_pos, "UH")
df["VB"]=count_pos(new_pos, "VB")
df["VBD"]=count_pos(new_pos, "VBD")
df["VBG"]=count_pos(new_pos, "VBG")
df["VBN"]=count_pos(new_pos, "VBN")
df["VBP"]=count_pos(new_pos, "VBP")
df["VBZ"]=count_pos(new_pos, "VBZ")
df["WDT"]=count_pos(new_pos, "WDT")
df["WP"]=count_pos(new_pos, "WP")
df["WP$"]=count_pos(new_pos, "WP$")
df["WRB"]=count_pos(new_pos, "WRB")


# In[36]:


pos_meta = df[['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
        'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
        'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB',
        'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']].values


# In[37]:


meta=np.concatenate((pos_meta, punc), axis=1)


# In[41]:


data.shape


# In[42]:


def A2Text():
    nlp_input = Input(shape=(327,), name='nlp_input')
    meta_input = Input(shape=(62,), name='meta_input')
    emb = Embedding(output_dim=EMB_DIM, input_dim=MAX_WORDS, input_length=MAX_SEQ_LENGTH)(nlp_input)
    flat_out = Flatten()(emb)
# nlp_out = Bidirectional(LSTM(16, dropout=0.3, recurrent_dropout=0.3,))(emb)
    x_con = Concatenate(axis=-1)([flat_out, meta_input])
    x_con=Dense(64, activation='relu')(x_con)
    x_con=Dropout(0.2)(x_con)
    x_con=Dense(1, activation='sigmoid')(x_con)
    # Output layer
    model_con = Model(inputs=[nlp_input , meta_input], outputs=[x_con])

  
    # print model summary
#     model.summary()
    
    # When using pretrained embeddings
    #model.layers[0].set_weights([embedding_matrix])
    #model.layers[0].trainable = False
              
    # Compile the model
    model_con.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['acc'])
    return model_con


# In[43]:


def precision_recall2(model, X_test, y_test):
    y_score = model.predict(X_test)
    y_score = np.array(y_score)
    y_score[y_score>0.5] = 1
    y_score[y_score<0.5] = 0
        
    precision, recall,f1, _ = precision_recall_fscore_support(y_test, y_score,average='weighted',labels=np.unique(y_score))
#     average_precision = average_precision_score(y_test.ravel(), y_score.ravel())
    return precision, recall,f1, _


# In[44]:


X=np.concatenate((data, meta), axis=1)
X.shape


# In[45]:


Dense_FPR = []

Dense_TPR = []

Dense_ROC = []

Dense_Recall = []

Dense_f1 = []
tprs= []

Dense_Precision=[]
# dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(fcmod,x_val,y_val)
# lst_fpr,lst_tpr,lst_roc_auc = TPR_FPR(lsmod, x_val, y_val)
# conv_fpr,conv_tpr,conv_roc_auc  = TPR_FPR(convmod, x_val, y_val)
# congru_fpr,congru_tpr,congru_roc_auc  = TPR_FPR(convgrumod, x_val, y_val)
# log_fpr,log_tpr,log_auc=TPR_FPR(log,x_val,y_val)


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)


kf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
for train_index, test_index in kf.split(X,labels):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train_nlp=X_train[:,0: 327]
    X_train_meta=X_train[:,327:389]
    X_test_nlp=X_test[:,0: 327]
    X_test_meta=X_test[:,327:389]
    a2tmod = A2Text()
    fchist= a2tmod.fit([X_train_nlp,X_train_meta], y_train,
         epochs = 10,
         batch_size = 256,
         validation_data = ([X_test_nlp,X_test_meta], y_test))
    
    dense_fpr,dense_tpr,dense_roc_auc = TPR_FPR(a2tmod,[X_test_nlp,X_test_meta], y_test)
    precision, recall,f1,_ = precision_recall2(a2tmod,[X_test_nlp,X_test_meta], y_test)
    tprs.append(interp(mean_fpr, dense_fpr, dense_tpr))
    Dense_Recall.append(recall)
    Dense_f1.append(f1)
    Dense_Precision.append(precision)
    Dense_FPR.append(dense_fpr)
    Dense_TPR.append(dense_tpr)
    Dense_ROC.append(dense_roc_auc)


# In[46]:


def Average(lst): 
    return sum(lst) / len(lst)

print("The Five-Cross Validation Average F1 Score is " , Average(Dense_f1))
print("The Five-Cross Validation Average Precesion Score is " , Average(Dense_Precision))
print("The Five-Cross Validation Average Recall Score is " , Average(Dense_Recall))
print("The Five-Cross Validation Average ROC Score is " , Average(Dense_ROC))






