import lime
from lime.lime_text import LimeTextExplainer
import os, pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import nltk, re,string
# downloading stopwords package
nltk.download('stopwords')
from nltk.corpus import stopwords
import en_core_web_sm

nlp = en_core_web_sm.load()
# removing english stopwords to save space and processing time
stop = stopwords.words('english')
class_names=['fake','real']
explainer=LimeTextExplainer(class_names=class_names)

with open(r"tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)
model = load_model('my_model.h5')

#def lemmatizer(text):
#    doc = nlp(text)
#    lemma_list = []
#    for token in doc:
#        if len(token) > 1:
#            lemma_list.append(token.lemma_.lower())
#    updated_text = " ".join(lemma_list)
#    return updated_text

def text_cleaning(text):
    '''making text lowercase, extracting 'reuters' tag from articles, removing text in square brackets,
    removiing all links, removing punctuation and removing words containing numbers.'''
    text = str(text).lower()
    text = re.sub('reuters', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    #remove special character
    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    text = ' '.join([word for word in text.split() if word not in (stop)])
    #text = lemmatizer(text)
    return text

def predict(text):
    processed = []
    processed.append(text_cleaning(text))
    tokenized = tokenizer.texts_to_sequences(processed)
    maxlen =1000
    tokenized = pad_sequences(tokenized, maxlen = maxlen)
    prediction = model.predict(tokenized)
    percent = prediction[0][0]
    percent = round(percent*100, 4)
    ans = (prediction >= 0.5).astype(int)
    return ans,percent
    

def new_predict(text):
    processed = []
    for i in text:
        processed.append(text_cleaning(i))
        #processed.append(i)
    tokenized = tokenizer.texts_to_sequences(processed)
    maxlen =1000
    tokenized = pad_sequences(tokenized, maxlen = maxlen)
    pred = model.predict(tokenized)
    pos_neg_preds = []
    for i in pred:
        temp=i[0]
        pos_neg_preds.append(np.array([1-temp,temp])) #I would recommend rounding temp and 1-temp off to 2 places
    return np.array(pos_neg_preds)

def text_predict_explain(content):
    explainer.explain_instance(content,new_predict,num_features=6).save_to_file('templates/text_explain.html')
    print("Saved")