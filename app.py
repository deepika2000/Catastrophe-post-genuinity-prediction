from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import *

def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def stemming(text):
    snow=SnowballStemmer("english")
    words=""
    for word in text:
        words+=snow.stem(word)+" "
    return words

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    train = pd.read_csv("train.csv", encoding="latin-1")
    features = train['text'].apply(text_process)
    features = features.apply(stemming)
    cv = CountVectorizer(max_features=1500)
    x = cv.fit_transform(features).toarray()
    y = train.iloc[:, 4].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test,y_test)
    #joblib.dump(clf, 'NB_genuine_model.pkl')
    #NB_genuine_model = open('NB_genuine_model.pkl','rb')
    #clf = joblib.load(NB_genuine_model)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
