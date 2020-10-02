from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from stop_words import get_stop_words
nltk.download('stopwords')
# Construction de la liste des stop words french
from nltk.stem import SnowballStemmer
fr = SnowballStemmer('french')
import re
from unidecode import unidecode

def NLPPredict(phrase):

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("data/feature.pkl", "rb")))
    user = transformer.fit_transform(loaded_vec.fit_transform([nettoyage(phrase)]))

    cls = pickle.load(open("data/cls.pkl", "rb"))

    return cls.predict(user)

def nettoyage(string, stopWord = None):
    l = []
    string = unidecode(string.lower())
    string = " ".join(re.findall("[a-zA-Z]+", string))

    if stopWord is None:
        for word in string.split():
            l.append(fr.stem(word))
    else:
        for word in string.split():
            if word in stopWord:
                continue
            else:
                l.append(fr.stem(word))
    return ' '.join(l)

def NLPTrain(file = 'data/corpus.csv'):
    df = pd.read_csv(file)
    df['l_review'] = df['review'].apply(lambda x: len(x.split(' ')))
    df = df[df['l_review'] > 5] #il faut au minimum 5 mots
    df['label'] = df['rating']
    nb_pos = len(df[df['label'] > 3])
    nb_neg = len(df[df['label'] < 3])
    nb_sample = nb_pos if nb_pos < nb_neg else nb_neg
    Corpus = pd.concat([df[df['label'] > 3].sample(nb_sample), df[df['label'] < 3].sample(nb_sample)], ignore_index=True)[['review', 'label']]
    del df
    my_stop_word_list = get_stop_words('french')
    final_stopwords_list = stopwords.words('french')
    s_w = list(set(final_stopwords_list + my_stop_word_list))
    s_w = [elem.lower() for elem in s_w]
    Corpus['label'] = Corpus['label'].apply(lambda x: 0 if x < 3 else 1)
    Corpus['review'] = Corpus['review'].apply(lambda x: nettoyage(x, s_w))
    vectorizer = TfidfVectorizer()
    vectorizer.fit(Corpus['review'])
    X = vectorizer.transform(Corpus['review'])
    pickle.dump(vectorizer.vocabulary_, open("data/feature.pkl","wb"))
    y = Corpus['label']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    cls = LogisticRegression(max_iter=300).fit(x_train, y_train)
    pickle.dump(cls, open("data/cls.pkl", "wb"))
    return cls.score(x_val, y_val)