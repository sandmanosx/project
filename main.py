# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
import copy
import json
import numpy as np
import pandas as pd
import nltk
import textacy
import re
import emoji
from textacy.preprocessing.replace import replace_hashtags
import string
from string import digits
from textacy.preprocessing.replace import replace_urls

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io
from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
import multiprocessing
import text2emotion as te
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

with open(r'CCMR_Google.txt', 'r') as file_open:
    data = json.load(file_open)
with open(r'CCMR_Twitter.txt', 'r') as file_open:
    data2 = json.load(file_open)

base = pd.DataFrame(data)
base2 = pd.DataFrame(data2)
processed1 = base
processed2 = base2
print(processed1)


#processed2.to_csv('twitter.csv', sep=',', header=True, index=True,encoding='utf-8')


def deconstructed(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def containenglish(str0):
    import re
    return bool(re.search('[a-z]', str0))


event = ['sandy', 'boston', 'sochi', 'bringback', 'malaysia', 'columbianChemicals', 'passport'
    , 'underwater', 'livr', 'pigFish', 'eclipse', 'garissa', 'nepal', 'samurai', 'syrianboy', 'varoufakis']

# PRE-PROCESSING######################################################################################################
######################################################################################################################

# DATA1 -----------------------------------------------------------------------------------------------------------------
# count hash tag,url,@
processed1['n_hash'] = 0
processed1['n_@'] = 0
processed1['n_url'] = 0

for index in range(len(processed1)):
    counts = processed1['title'][index].count('#')
    processed1['n_hash'][index] = counts
    counts = processed1['title'][index].count('@')
    processed1['n_@'][index] = counts
    counts = processed1['title'][index].count('http')
    processed1['n_url'][index] = counts

# remove appendix
print('removing appendix...')
for i in range(10):
    result = base['title'].apply(lambda x: x.split('-')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('–')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('–')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('—')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('|')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('http')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('(')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('[')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('/')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('>')[0])
    processed1['title'] = result
    result = processed1['title'].apply(lambda x: x.split('»')[0])
    processed1['title'] = result

# remove emoji
print('removing emoji....')
for index in range(len(processed1)):
    demoji = emoji.demojize(processed1['title'][index])
    result = re.sub(':\S+?:', ' ', demoji)
    processed1['title'][index] = result

# decontract I'm->I am
print('decontracting....')
for index in range(len(processed1)):
    result = deconstructed(processed1['title'][index])
    processed1['title'][index] = result

# remove hash tags
print('removing hash tags....')
for index in range(len(processed1)):
    result = replace_hashtags(processed1['title'][index], replace_with='')
    processed1['title'][index] = result

# remove @+name
print('removing @+name....')
for index in range(len(processed1)):
    result = re.sub('@[^\s]+', '', processed1['title'][index])
    processed1['title'][index] = result

# all words to lowercase
print('translating all words to lower cass ....')
for index in range(len(processed1)):
    result = processed1['title'][index].lower()
    processed1['title'][index] = result

# remove number
print('removing number....')
for index in range(len(processed1)):
    result = processed1['title'][index].translate(str.maketrans('', '', digits))
    processed1['title'][index] = result

# Remove whitespace and punctuation
print('removing whitespace and punctuation....')
for index in range(len(processed1)):
    result = re.sub(r'[^A-Za-z0-9 ]+', '', processed1['title'][index])
    processed1['title'][index] = result

# remove non-english words
print('removing non-english words')
words = set(nltk.corpus.words.words())
for index in range(len(processed1)):
    result = " ".join(w for w in nltk.wordpunct_tokenize(processed1['title'][index]) \
                      if w.lower() in words or not w.isalpha())
    processed1['title'][index] = result

#processed1.to_csv('processed1.csv', sep=',', header=True, index=True)

# data 2--------------------------------------------------------------------------------------------------------
# count hash tag,url,@
processed2['n_hash'] = 0
processed2['n_@'] = 0
processed2['n_url'] = 0
for index in range(len(processed2)):
    counts = processed2['content'][index].count('#')
    processed2['n_hash'][index] = counts
    counts = processed2['content'][index].count('@')
    processed2['n_@'][index] = counts
    counts = processed2['content'][index].count('http')
    processed2['n_url'][index] = counts

# remove emoji
print('removing emoji....')
for index in range(len(base2)):
    demoji = emoji.demojize(base2['content'][index])
    result = re.sub(':\S+?:', ' ', demoji)
    processed2['content'][index] = result

# decontract
print('decontracting')
for index in range(len(processed2)):
    result = deconstructed(processed2['content'][index])
    processed2['content'][index] = result

# delete URL
print('removing URL....')
for index in range(len(processed2)):
    result = replace_urls(processed2['content'][index], replace_with='')
    processed2['content'][index] = result

# remove hash tags
print('removing hash tags....')
for index in range(len(processed2)):
    result = replace_hashtags(processed2['content'][index], replace_with='')
    processed2['content'][index] = result

# Replace currency symbol
print('replacing currency symbol....')
for index in range(len(processed2)):
    result = textacy.preprocessing.replace.replace_currency_symbols(processed2['content'][index])
    processed2['content'][index] = result

# remove full capitalized words
print('removing full acpitalized words....')
for index in range(len(processed2)):
    result = re.sub(r'\b[A-Z]+\b', '', processed2['content'][index])
    processed2['content'][index] = result

# remove @+name
print('replacing @+name....')
for index in range(len(processed2)):
    result = re.sub('@[^\s]+', '', processed2['content'][index])
    processed2['content'][index] = result

# all words to lowercase
print('all words to lowercase')
for index in range(len(processed2)):
    result = processed2['content'][index].lower()
    processed2['content'][index] = result

# remove number
print('removing number....')
for index in range(len(processed1)):
    result = processed2['content'][index].translate(str.maketrans('', '', digits))
    processed2['content'][index] = result

# Remove whitespace and punctuation"
print('removing whitespace and punctuation....')
for index in range(len(processed2)):
    result = re.sub(r'[^A-Za-z0-9 ]+', '', processed2['content'][index])
    processed2['content'][index] = result

# remove non-english words
print('removing non-english words....')
words = set(nltk.corpus.words.words())
for index in range(len(processed2)):
    result = " ".join(w for w in nltk.wordpunct_tokenize(processed2['content'][index]) \
                      if w.lower() in words or not w.isalpha())
    processed2['content'][index] = result

# processed2.to_csv('processed2.csv', sep=',', header=True, index=True)


# SPLIT & VECTORIZATION#################################################################################################
#######################################################################################################################


# split&mark the speech
processed1['splited'] = 0
processed2['splited'] = 0

for index in range(len(processed1)):
    processed1['splited'][index] = nltk.word_tokenize(processed1['title'][index])

for index in range(len(processed2)):
    processed2['splited'][index] = nltk.word_tokenize(processed2['content'][index])

# remove stop words
processed1['final'] = 0
processed2['final'] = 0
# stop_words = set(stopwords.words('english'))
for index in range(len(processed1)):
    processed1['final'][index] = [word for word in processed1['splited'][index] if
                                  word not in stopwords.words('english')]

for index in range(len(processed2)):
    processed2['final'][index] = [word for word in processed2['splited'][index] if
                                  word not in stopwords.words('english')]

##################################Delete missing values############################
for i in range(len(processed1)):
    if (len(processed1['final'][i]) == 0):
        processed1.drop(i, inplace=True)

for i in range(len(processed2)):
    if (len(processed2['final'][i]) == 0):
        processed2.drop(i, inplace=True)

processed1 = processed1.reset_index(drop=True)
processed2 = processed2.reset_index(drop=True)

# split data by topic
google_split = []
twitter_split = []
google_num = []
twitter_num = []
num_array = []
num = 0

for i in range(16):
    for j in range(len(processed1)):
        if processed1['event'][j] == event[i]:
            num_array.append(j)
            num = num + 1
    google_split.append(num)
    # google_split : every events' cases' number respectively
    google_num.append(num_array.copy())
    # google_num : the cases' location number respectively
    num_array = []
    num = 0

for i in range(16):
    for j in range(len(processed2)):
        if processed2['event'][j] == event[i]:
            num_array.append(j)
            num = num + 1
    twitter_split.append(num)
    twitter_num.append(num_array.copy())
    num_array = []
    num = 0

# reunit to long string
united = ''
processed1['united'] = 0
processed2['united'] = 0

# add united to refresh every raw of the dataset
for i in range(len(processed1)):
    for j in range(len(processed1['final'][i])):
        united = united + ' ' + processed1['final'][i][j]
    processed1['united'][i] = united
    united = ''

for i in range(len(processed2)):
    for j in range(len(processed2['final'][i])):
        united = united + ' ' + processed2['final'][i][j]
    processed2['united'][i] = united
    united = ''

# emotion analyse
google_emotion = []
for i in range(len(processed1)):
    text = processed1['title'][i]
    google_emotion.append(te.get_emotion(text))

twitter_emotion = []
for i in range(len(processed2)):
    text = processed2['content'][i]
    twitter_emotion.append(te.get_emotion(text))

# vectorization
model_google = {}
model_twitter = {}
model_cross = {}
certain_title = []
tfidf_google = {}
tfidf_twitter = {}
tfidf_cross = {}

for i in range(16):
    for index in range(len(google_num[i])):
        certain_title.append(processed1['united'][google_num[i][index]])
        # add the processed united string respectively
    model = TfidfVectorizer()
    re = model.fit_transform(certain_title)
    tfidf_google[i] = re
    model_google[i] = model
    certain_title = []

for i in range(16):
    for index in range(len(twitter_num[i])):
        certain_title.append(processed2['united'][twitter_num[i][index]])
    model = TfidfVectorizer()
    re = model.fit_transform(certain_title)
    tfidf_twitter[i] = re
    model_twitter[i] = model
    certain_title = []

for i in range(16):
    for index in range(len(google_num[i])):
        certain_title.append(processed1['united'][google_num[i][index]])
    for index in range(len(twitter_num[i])):
        certain_title.append(processed2['united'][twitter_num[i][index]])
    model = TfidfVectorizer()
    re = model.fit_transform(certain_title)
    tfidf_cross[i] = re
    model_cross[i] = model
    certain_title = []

# print(tfidf_google[9])
# print(tfidf_twitter[9])
# print(tfidf_cross[9])
# print("!!!!!!!!!!!!!!!!!!!!!!!!!")

# Feature_extract#######################################################################################################
#######################################################################################################################
# get array
google_array = {}
twitter_array = {}
cross_array = {}
for i in range(16):
    cross_array[i] = tfidf_cross[i].toarray()
    google_array[i] = tfidf_google[i].toarray()
    twitter_array[i] = tfidf_twitter[i].toarray()
# print(google_array[11])
# calculate similarity COS
google_cos = {}
twitter_cos = {}
cross_cos = {}
for i in range(16):
    google_cos[i] = cosine_similarity(google_array[i])
    twitter_cos[i] = cosine_similarity(twitter_array[i])
    cross_cos[i] = cosine_similarity(cross_array[i])
# print(google_cos[11])

####################################plot heatmap##################################################
##################################################################################################
for index in range(16):
    sns.set()
    ax = sns.heatmap(google_cos[index], vmin=0, vmax=1)
    plt.savefig('google' + str(index) + '.png')
    plt.show()
    plt.close()
    sns.set()
    ax = sns.heatmap(twitter_cos[index], vmin=0, vmax=1)
    plt.savefig('twitter' + str(index) + '.png')
    plt.show()
    plt.close()
    ax = sns.heatmap(cross_cos[index], vmin=0, vmax=1)
    plt.savefig('cross' + str(index) + '.png')
    plt.show()
    plt.close()

#####################MACHINE LEARNING###################################################################################
########################################################################################################################
del base, base2, cross_cos, twitter_cos, google_cos, data, data2, demoji, digits, file_open, model, model_google, \
    model_twitter, model_cross, re, stopwords, tfidf_cross, tfidf_google, tfidf_twitter, words


# multiprocessing.set_start_method('forkserver')


def feature_excel_init(feature_pd):
    feature_pd = pd.DataFrame(columns=['impor_happy', 'impor_angry', 'impor_surprise', 'impor_sad' \
        , 'impor_fear', 'impor_emotion', 'impor_hash', 'impor_@', 'impor_url', 'impor_vector' \
        , 'n_vector'])

    return feature_pd


def write_feature(classifitor, feature_pd, indici):
    importance = classifitor.feature_importances_
    vec_impor = []
    vec_emotion = []
    # add vectors' importances from 8th on
    for i in range(8, len(importance)):
        vec_impor.append(importance[i])
    for i in range(5):
        vec_emotion.append(importance[i])
    total_emotion = np.sum(vec_emotion)
    total_vec_impor = np.sum(vec_impor)
    len_vec = len(importance) - 8
    feature_pd.loc[indici, 'impor_happy'] = importance[0]
    feature_pd.loc[indici, 'impor_angry'] = importance[1]
    feature_pd.loc[indici, 'impor_surprise'] = importance[2]
    feature_pd.loc[indici, 'impor_sad'] = importance[3]
    feature_pd.loc[indici, 'impor_fear'] = importance[4]
    feature_pd.loc[indici, 'impor_emotion'] = total_emotion
    feature_pd.loc[indici, 'impor_hash'] = importance[5]
    feature_pd.loc[indici, 'impor_@'] = importance[6]
    feature_pd.loc[indici, 'impor_url'] = importance[7]
    feature_pd.loc[indici, 'impor_vector'] = total_vec_impor
    feature_pd.loc[indici, 'n_vector'] = len_vec
    return feature_pd


def write_result(result_pd, method, y_predicted):
    result_pd[method] = y_predicted.tolist()
    return result_pd


# test = write_result(names['googleresult_%s'%index],'RF',y_pre)
#
# google_feature = write_feature(clf, google_feature, '_RF', index)

# google================================================================================================================
google_feature = pd.DataFrame()
google_feature = feature_excel_init(google_feature)

names = locals()
for i in range(16):
    names['googleresult_%s' % i] = pd.DataFrame(columns=['origin', 'NB', 'RF', 'SVM', 'TPOT'])

for index in range(16):
    print('now processing Google: ', index)
    X = pd.DataFrame()
    Y = pd.DataFrame()
    N_H = pd.DataFrame()
    N_A = pd.DataFrame()
    N_U = pd.DataFrame()
    # hash, @,url
    N_happy = pd.DataFrame()
    N_angry = pd.DataFrame()
    N_surprise = pd.DataFrame()
    N_sad = pd.DataFrame()
    N_fear = pd.DataFrame()
    n_h = []
    n_a = []
    n_u = []
    n_happy = []
    n_angry = []
    n_surprise = []
    n_sad = []
    n_fear = []
    Xlabel = []
    for i in range(len(google_num[index])):
        Xlabel.append(processed1['label'][google_num[index][i]])
    for i in range(len(google_num[index])):
        n_happy.append(google_emotion[google_num[index][i]]['Happy'])
        n_angry.append(google_emotion[google_num[index][i]]['Angry'])
        n_surprise.append(google_emotion[google_num[index][i]]['Surprise'])
        n_sad.append(google_emotion[google_num[index][i]]['Sad'])
        n_fear.append(google_emotion[google_num[index][i]]['Fear'])
        n_h.append(processed1['n_hash'][google_num[index][i]])
        n_a.append(processed1['n_@'][google_num[index][i]])
        n_u.append(processed1['n_url'][google_num[index][i]])
    X = google_array[index]
    N_H = n_h
    N_A = n_a
    N_U = n_u
    N_happy = n_happy
    N_angry = n_angry
    N_surprise = n_surprise
    N_sad = n_sad
    N_fear = n_fear
    X = np.insert(X, 0, values=N_U, axis=1)
    X = np.insert(X, 0, values=N_A, axis=1)
    X = np.insert(X, 0, values=N_H, axis=1)
    X = np.insert(X, 0, values=N_fear, axis=1)
    X = np.insert(X, 0, values=N_sad, axis=1)
    X = np.insert(X, 0, values=N_surprise, axis=1)
    X = np.insert(X, 0, values=N_angry, axis=1)
    X = np.insert(X, 0, values=N_happy, axis=1)
    Y = np.array(Xlabel)
    ################################scikit-learn###############################
    x_train, x_test, y_train, y_test = train_test_split(X.astype(np.float64),
                                                        Y, train_size=0.75,
                                                        test_size=0.25)

    names['googleresult_%s' % index] = write_result(names['googleresult_%s' % index], 'origin', y_test)
    try:
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of google:', index, ' (NB)')
        report = classification_report(y_test, y_pre, target_names=['0', '1', '2'], digits=7)
        print(report)
        try:
            names['googleresult_%s' % index] = write_result(names['googleresult_%s' % index], 'NB', y_pre)
        except:
            print('no result of google' + str(index) + '_NB')
    except:
        print('google ', index, ' NB error')
    try:
        clf = RandomForestClassifier(random_state=42)
        # clf = RandomForestClassifier(n_estimators=21, max_depth=None, random_state=42)
        # score_pre = cross_val_score(clf, data.data, data.target, cv=10).mean()
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of google:', index, ' (RF)')
        report = classification_report(y_test, y_pre, digits=7)
        print(report)
        google_feature = write_feature(clf, google_feature, index)
        names['googleresult_%s' % index] = write_result(names['googleresult_%s' % index], 'RF', y_pre)
    except:
        print('google ', index, ' RF error')
    try:
        clf = svm.SVC(kernel='linear', random_state=42)
        clf.fit(x_train, y_train)
        # w = clf.coef_[0]
        # a = - w[0] / w[1]
        # xx = np.linspace(-5, 5)
        # yy = a * xx - (clf.intercept_[0] / w[1])
        #
        # b = clf.support_vectors_[0]
        # yy_down = a * xx + (b[1] - a * b[0])
        #
        # b = clf.support_vectors_[-1]
        # yy_up = a * xx + (b[1] - a * b[0])
        #
        # print("w: ", w)
        # print("a: ", a)
        # print("support_vectors_: ", clf.support_vectors_)
        # print("clf.coef_: ", clf.coef_)
        # plt.plot(xx, yy, 'k-')
        # plt.plot(xx, yy_down, 'k--')
        # plt.plot(xx, yy_up, 'k--')
        # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
        # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
        # plt.axis('tight')
        # plt.show()

        y_pre = clf.predict(x_test)
        print('result of google:', index, ' (SVM)')
        report = classification_report(y_test, y_pre, target_names=['0', '1', '2'], digits=7)
        print(report)
        names['googleresult_%s' % index] = write_result(names['googleresult_%s' % index], 'SVM', y_pre)
    except:
        print('google ', index, ' SVM error')
    try:
        pipeline_optimizer = TPOTClassifier(generations=30, population_size=100, cv=5,
                                            random_state=42, verbosity=2, use_dask=True)
        print('result of google:', index, ' (TPOT)')
        pipeline_optimizer.fit(x_train, y_train)
        y_pre = pipeline_optimizer.predict(x_test)
        report = classification_report(y_test, y_pre, target_names=['0', '1', '2'], digits=7)
        print(report)
        names['googleresult_%s' % index] = write_result(names['googleresult_%s' % index], 'TPOT', y_pre)
    except:
        print('google ', index, ' TPOT error')

writer = pd.ExcelWriter('google_feature.xlsx')
google_feature.to_excel(writer, sheet_name='google_feature')
writer.save()
writer.close()

writer = pd.ExcelWriter('google_result.xlsx')
use = [0, 1, 2, 3, 4, 10, 12]
for index in use:
    names['googleresult_%s' % index].to_excel(writer, sheet_name='google' + str(index + 1))
writer.save()
writer.close()

del google_feature
for i in range(16):
    del names['googleresult_%s' % i]
# twitter===============================================================================================================
twitter_feature = pd.DataFrame()
twitter_feature = feature_excel_init(twitter_feature)
names = locals()
for i in range(16):
    names['twitter_result_%s' % i] = pd.DataFrame(columns=['origin', 'NB', 'RF', 'SVM', 'TPOT'])

for index in range(16):
    print('now processing Twitter: ')
    print(index)
    X = pd.DataFrame()
    Y = pd.DataFrame()
    N_H = pd.DataFrame()
    N_A = pd.DataFrame()
    N_U = pd.DataFrame()
    N_happy = pd.DataFrame()
    N_angry = pd.DataFrame()
    N_surprise = pd.DataFrame()
    N_sad = pd.DataFrame()
    N_fear = pd.DataFrame()
    n_h = []
    n_a = []
    n_u = []
    n_happy = []
    n_angry = []
    n_surprise = []
    n_sad = []
    n_fear = []
    Xlabel = []
    for i in range(len(twitter_num[index])):
        Xlabel.append(processed2['label'][twitter_num[index][i]])
    for i in range(len(twitter_num[index])):
        n_h.append(processed2['n_hash'][twitter_num[index][i]])
        n_a.append(processed2['n_@'][twitter_num[index][i]])
        n_u.append(processed2['n_url'][twitter_num[index][i]])
        n_happy.append(twitter_emotion[twitter_num[index][i]]['Happy'])
        n_angry.append(twitter_emotion[twitter_num[index][i]]['Angry'])
        n_surprise.append(twitter_emotion[twitter_num[index][i]]['Surprise'])
        n_sad.append(twitter_emotion[twitter_num[index][i]]['Sad'])
        n_fear.append(twitter_emotion[twitter_num[index][i]]['Fear'])
    X = twitter_array[index]
    N_H = n_h
    N_A = n_a
    N_U = n_u
    N_happy = n_happy
    N_angry = n_angry
    N_surprise = n_surprise
    N_sad = n_sad
    N_fear = n_fear
    X = np.insert(X, 0, values=N_U, axis=1)
    X = np.insert(X, 0, values=N_A, axis=1)
    X = np.insert(X, 0, values=N_H, axis=1)
    X = np.insert(X, 0, values=N_fear, axis=1)
    X = np.insert(X, 0, values=N_sad, axis=1)
    X = np.insert(X, 0, values=N_surprise, axis=1)
    X = np.insert(X, 0, values=N_angry, axis=1)
    X = np.insert(X, 0, values=N_happy, axis=1)
    Y = np.array(Xlabel)

    x_train, x_test, y_train, y_test = train_test_split(X.astype(np.float64), Y, train_size=0.75,
                                                        test_size=0.25)
    names['twitter_result_%s' % index] = write_result(names['twitter_result_%s' % index], 'origin', y_test)
    try:
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of twitter:', index, ' (NB)')
        report = classification_report(y_test, y_pre, target_names=['0', '1'], digits=7)
        print(report)
        names['twitter_result_%s' % index] = write_result(names['twitter_result_%s' % index], 'NB', y_pre)
    except:
        print('twitter ', index, ' NB error')
    try:
        clf = RandomForestClassifier(random_state=42)
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of twitter:', index, ' (RF)')
        report = classification_report(y_test, y_pre, digits=7)
        print(report)
        twitter_feature = write_feature(clf, twitter_feature, index)
        names['twitter_result_%s' % index] = write_result(names['twitter_result_%s' % index], 'RF', y_pre)
    except:
        print('twitter ', index, ' RF error')
    try:
        clf = svm.SVC(kernel='linear', random_state=42)
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of twitter:', index, ' (SVM)')
        report = classification_report(y_test, y_pre, target_names=['0', '1'], digits=7)
        print(report)
        names['twitter_result_%s' % index] = write_result(names['twitter_result_%s' % index], 'SVM', y_pre)
    except:
        print('twitter ', index, ' SVM error')
    try:
        pipeline_optimizer = TPOTClassifier(generations=3, population_size=10, cv=5,
                                            random_state=42, verbosity=2, use_dask=True)
        print('result of twitter:', index, ' (TPOT)')
        pipeline_optimizer.fit(x_train, y_train)
        y_pre = pipeline_optimizer.predict(x_test)
        report = classification_report(y_test, y_pre, target_names=['0', '1'], digits=7)
        print(report)
        names['twitter_result_%s' % index] = write_result(names['twitter_result_%s' % index], 'TPOT', y_pre)
    except:
        print('twitter ', index, ' TPOT error')

writer = pd.ExcelWriter('twitter_feature.xlsx')
twitter_feature.to_excel(writer, sheet_name='twitter_feature')
writer.save()
writer.close()

writer = pd.ExcelWriter('twitter_result.xlsx')
use = [0, 1, 10, 12]
for index in use:
    names['twitter_result_%s' % index].to_excel(writer, sheet_name='twitter' + str(index + 1))
writer.save()
writer.close()

del twitter_feature
for i in range(16):
    del names['twitter_result_%s' % i]
# CROSS=================================================================================================================
cross_feature = pd.DataFrame()
cross_feature = feature_excel_init(cross_feature)
names = locals()
for i in range(16):
    names['cross_result_%s' % i] = pd.DataFrame(columns=['origin', 'NB', 'RF', 'SVM', 'TPOT'])

for index in range(16):
    print('now processing CROSS: ')
    print(index)
    X = pd.DataFrame()
    Y = pd.DataFrame()
    N_H = pd.DataFrame()
    N_A = pd.DataFrame()
    N_U = pd.DataFrame()
    N_happy = pd.DataFrame()
    N_angry = pd.DataFrame()
    N_surprise = pd.DataFrame()
    N_sad = pd.DataFrame()
    N_fear = pd.DataFrame()
    n_h = []
    n_a = []
    n_u = []
    n_happy = []
    n_angry = []
    n_surprise = []
    n_sad = []
    n_fear = []
    Xlabel = []
    for i in range(len(google_num[index])):
        Xlabel.append(processed1['label'][google_num[index][i]])
    for i in range(len(twitter_num[index])):
        Xlabel.append(processed2['label'][twitter_num[index][i]])
    for i in range(len(google_num[index])):
        n_h.append(processed1['n_hash'][google_num[index][i]])
        n_a.append(processed1['n_@'][google_num[index][i]])
        n_u.append(processed1['n_url'][google_num[index][i]])
        n_happy.append(google_emotion[google_num[index][i]]['Happy'])
        n_angry.append(google_emotion[google_num[index][i]]['Angry'])
        n_surprise.append(google_emotion[google_num[index][i]]['Surprise'])
        n_sad.append(google_emotion[google_num[index][i]]['Sad'])
        n_fear.append(google_emotion[google_num[index][i]]['Fear'])
    for i in range(len(twitter_num[index])):
        n_h.append(processed2['n_hash'][twitter_num[index][i]])
        n_a.append(processed2['n_@'][twitter_num[index][i]])
        n_u.append(processed2['n_url'][twitter_num[index][i]])
        n_happy.append(twitter_emotion[twitter_num[index][i]]['Happy'])
        n_angry.append(twitter_emotion[twitter_num[index][i]]['Angry'])
        n_surprise.append(twitter_emotion[twitter_num[index][i]]['Surprise'])
        n_sad.append(twitter_emotion[twitter_num[index][i]]['Sad'])
        n_fear.append(twitter_emotion[twitter_num[index][i]]['Fear'])
    X = cross_array[index]
    N_H = n_h
    N_A = n_a
    N_U = n_u
    N_happy = n_happy
    N_angry = n_angry
    N_surprise = n_surprise
    N_sad = n_sad
    N_fear = n_fear
    X = np.insert(X, 0, values=N_U, axis=1)
    X = np.insert(X, 0, values=N_A, axis=1)
    X = np.insert(X, 0, values=N_H, axis=1)
    X = np.insert(X, 0, values=N_fear, axis=1)
    X = np.insert(X, 0, values=N_sad, axis=1)
    X = np.insert(X, 0, values=N_surprise, axis=1)
    X = np.insert(X, 0, values=N_angry, axis=1)
    X = np.insert(X, 0, values=N_happy, axis=1)
    Y = np.array(Xlabel)

    x_train, x_test, y_train, y_test = train_test_split(X.astype(np.float64), Y, train_size=0.75,
                                                        test_size=0.25)
    names['cross_result_%s' % index] = write_result(names['cross_result_%s' % index], 'origin', y_test)

    try:
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of CROSS:', index, ' (NB)')
        names['cross_result_%s' % index] = write_result(names['cross_result_%s' % index], 'NB', y_pre)
        report = classification_report(y_test, y_pre, target_names=['0', '1', '2'], digits=7)
        print(report)
    except:
        print('CROSS ', index, ' NB error')
    try:
        clf = RandomForestClassifier(random_state=42)
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of CROSS:', index, ' (RF)')
        report = classification_report(y_test, y_pre, digits=7)
        print(report)
        cross_feature = write_feature(clf, cross_feature, index)
        names['cross_result_%s' % index] = write_result(names['cross_result_%s' % index], 'RF', y_pre)

    except:
        print('CROSS ', index, ' RF error')
    try:
        clf = svm.SVC(kernel='linear', random_state=42)
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        print('result of CROSS:', index, ' (SVM)')
        report = classification_report(y_test, y_pre, target_names=['0', '1', '2'], digits=7)
        print(report)
        names['cross_result_%s' % index] = write_result(names['cross_result_%s' % index], 'SVM', y_pre)
    except:
        print('CROSS ', index, ' SVM error')
    try:
        pipeline_optimizer = TPOTClassifier(generations=3, population_size=10, cv=5,
                                            random_state=42, verbosity=2, use_dask=True)
        print('result of CROSS:', index, ' (TPOT)')
        pipeline_optimizer.fit(x_train, y_train)
        y_pre = pipeline_optimizer.predict(x_test)
        report = classification_report(y_test, y_pre, target_names=['0', '1', '2'], digits=7)
        print(report)
        names['cross_result_%s' % index] = write_result(names['cross_result_%s' % index], 'TPOT', y_pre)
    except:
        print('CROSS ', index, ' TPOT error')

writer = pd.ExcelWriter('cross_feature.xlsx')
cross_feature.to_excel(writer, sheet_name='cross_feature')
writer.save()
writer.close()

writer = pd.ExcelWriter('cross_result.xlsx')
use = [0, 1, 2, 3, 4, 5, 9, 10, 12]
for index in use:
    names['cross_result_%s' % index].to_excel(writer, sheet_name='cross' + str(index + 1))
writer.save()
writer.close()

del cross_feature
for i in range(16):
    del names['cross_result_%s' % i]
