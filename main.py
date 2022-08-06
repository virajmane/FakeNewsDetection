import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import cross_origin
from ast import literal_eval
from nltk import RegexpTokenizer, PorterStemmer
from nltk.corpus import stopwords
import requests
import re

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
@cross_origin()
def main():
    if request.method=="POST":
        original = request.form.get('fname')
        if "http" in original:
            resp = trusteddomains(original)
            result = resp[0]
            status = resp[1]
        else:
            news = {"news_text":original}
            r = requests.post("https://24hrchallenge.virajman3.repl.co/predict", data=f'{news}')
            data = r.json()
            label = data["label"]
            probablity = int(float(data['probability'])*100)
            status = "danger"
            if label=="real":
                status = "success"
            result = f"The model is {probablity}% sure that the above news is {label}"
        return render_template("index.html", txt1=original, txt2=result, status = status)                    
    return render_template("index.html", txt1="", txt2="")



@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        news = request.data
        print(news)
        data = literal_eval(news.decode('utf8'))
        value = api(pd.DataFrame([data]))
        value = value.iloc[0]
        if value['label'] == 1:
            value['label'] = "fake"
            return value.to_dict()
        value['label'] = "real"
        return value.to_dict()


@app.route("/predict_list", methods=["GET", "POST"])
@cross_origin()
def predict_list():
    if request.method == "POST":
        news = request.data
        data = literal_eval(news.decode('utf8'))
        value = api(pd.DataFrame(data))
        value['label'] = np.where(value['label'] == 0, 'real', 'fake')
        return value.to_dict(orient='index')


def preprocess_data(data):
    # 1. Tokenization
    tk = RegexpTokenizer('\s+', gaps=True)
    text_data = []  # List for storing the tokenized data
    for values in data.news_text:
        tokenized_data = tk.tokenize(values)  # Tokenize the news
        text_data.append(tokenized_data)  # append the tokenized data

    # 2. Stopword Removal
    # Extract the stopwords
    sw = stopwords.words('english')
    clean_data = []  # List for storing the clean text
    # Remove the stopwords using stopwords
    for data in text_data:
        clean_text = [words.lower() for words in data if words.lower() not in sw]
        clean_data.append(clean_text)  # Appned the clean_text in the clean_data list

    # 3. Stemming
    # Create a stemmer object
    ps = PorterStemmer()
    stemmed_data = []  # List for storing the stemmed data
    for data in clean_data:
        stemmed_text = [ps.stem(words) for words in data]  # Stem the words
        stemmed_data.append(stemmed_text)  # Append the stemmed text
    updated_data = []
    for data in stemmed_data:
        updated_data.append(" ".join(data))
    return updated_data


def api(df_test):
    try:
        with open("fakenewsmodel.pkl", "rb") as file:
            model = pickle.load(file)
    except:
        print("Unable to load the Fake news model pickle file, please check the spelling and path")
    preprocessed_testdata = preprocess_data(df_test)
    try:
        with open("tfidfmodel.pkl", "rb") as file:
            tfidf = pickle.load(file)
    except:
        print("Unable to load the Tfid model pickle file, please check the spelling and path")
    preprocessed_testdata = tfidf.transform(preprocessed_testdata)
    features_df = pd.DataFrame(preprocessed_testdata.toarray())
    df_test["label"] = model.predict(features_df)
    probabs = model.predict_proba(features_df)
    probs = list()
    for prob in probabs:
        probs.append(round(max(prob[0], prob[1]), 2))
    df_test["probability"] = probs
    return df_test

def trusteddomains(name):
    trusteddomains =['5dariyanews.com', 'agranews.com', 'thenewsminute.com', 'pioneeredge.in', 'powersportz.tv', 'indianspectator.com', 'anytvnews.com', 'pressmirchi.com', 'altnews.in', 'chandigarhmetro.com', 'dailypioneer.com', 'jknewsline.com', 'newsblare.com', 'stardesk.in', 'thetimesofbengal.com', 'odishabarta.com', 'gudstory.com', 'navhindtimes.in', 'kashmirreader.com', 'newstodaynet.com', 'ways2rock.com', 'tentaran.com', 'knnindia.co.in', 'socialsamosa.com', 'latestnewsupdate4you.blogspot.com', 'hindi.webdunia.com', 'reddyrishvanth.wordpress.com', 'latestnewsworld24x7.blogspot.com', 'teluguglobal.in', 'assamtribune.com', 'seelatest.com', 'techgenyz.com', 'deccanherald.com', 'indiavision.com', 'news4masses.com', 'indvox.com', 'yovizag.com', 'kashmirobserver.net', 'firstpost.com', 'theprint.in', 'opindia.com', 'dkoding.in', 'thestatesman.com', 'anewsofindia.com', 'cms.abclive.in', 'timesofindia.indiatimes.com', 'indiareal.in', 'thehindu.com', 'ndtv.com', 'newsinonee.blogspot.com', 'indiasnews.net', 'thenorthlines.com', 'letmethink.in', 'livelaw.in', 'ndnewsexpress.com', 'informalnewz.com', 'headlinesoftoday.com', 'assamtimes.org', 'thesangaiexpress.com', 'theindiabizz.com', 'freepressjournal.in', 'smarttechtoday.com', 'news20today.blogspot.com', 'thebetterindia.com', 'nayanazriya.com', 'onmanorama.com', 'growideindia.com', 'sinceindependence.com', 'aajtakshweta.com', 'organiser.org', 'quintdaily.com', 'swachhindia.ndtv.com', 'siasat.com', 'indiatvnews.com', 'salahuddinayyubi.com', 'notabletoday.blogspot.com', 'reviewminute.com', 'newsdharm.blogspot.com', 'indianyug.com', 'thenewsglory.com', 'sahilonline.net', 'indiannewsnetwork.net', 'timesnowindia.com', 'abcrnews.com', 'starofmysore.com', 'oneindia.com', 'news1india.in', 'amarujala.com', 'news.abplive.com', 'english.jagran.com', 'thenewshimachal.com', 'india.com', 'thelivenagpur.com', 'newsdeets.com', 'topblogmania.com', 'elkeesmedia.com', 'livemint.com', 'emitpost.com', 'tfipost.com', 'doonhorizon.in', 'bangaloretoday.in', 'gonewsindia.com', 'theshillongtimes.com', 'democraticjagat.com', 'patrika.com', 'scroll.in', 'bhaskarlive.in', 'www.', 'hydnews.net', 'asianage.com', 'asian-times.com', 'paletv.com', 'deccanchronicle.com', 'editorji.com', 'crowdwisdom.live', 'indiaobservers.com', 'digpu.com', 'sentinelassam.com', 'goodnewwws.in', 'morungexpress.com', 'tribuneindia.com', 'business-standard.com', 'telanganatoday.com', 'greaterkashmir.com', 'thewire.in', 'newindianexpress.com', 'dailyexcelsior.com', 'krooknews.com', 'dnaindia.com', 'outlookindia.com', 'thenewsmill.com', 'apnlive.com', 'indianexpress.com', 'easternherald.com', 'bangaloremirror.indiatimes.com', 'delhipostnews.com', 'hellovizag.online', 'thehindubusinessline.com', 'orissapost.com', 'chandigarhcitynews.com', 'firstpostofindia.com', 'financialexpress.com', 'frontline.thehindu.com', 'pragativadi.com', 'web.statetimes.in', 'rediff.com', 'indiatoday.in', 'leagueofindia.com', 'nationalheraldindia.com', 'arunachaltimes.in', 'bharatsuchana.com', 'mobilenewspepar.in', 'startupreporter.in', 'oibnews.com', 'sportskanazee.com', 'news18.com', 'thekashmirwalla.com']
    domain = re.findall('^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)', name)
    if domain[0] in trusteddomains:
        return ["The domain name is trusted", "success"]
    else:
        return ["The domain name is not present in our list of trusted domains.", "danger"]

class FakeNewsApiService:
    def start(self):
        app.run(debug=True, use_reloader=False)


# run the api
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True, use_reloader=False)
