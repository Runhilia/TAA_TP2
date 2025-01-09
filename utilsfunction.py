from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import string
from sklearn.pipeline import Pipeline
import multiprocessing
from gensim.models import Word2Vec
import gensim

# Fonction pour préparer les données
def prepare_data(data):
    data_drop = data.drop(columns=['Title', 'meshMajor', 'pmid', "meshid", 'meshroot'])

    # Suppression des stopwords
    data_drop['abstractText'] = data_drop['abstractText'].apply(lambda x: remove_stopwords(x) if isinstance(x, str) else x)
    # Mise en minuscule
    data_drop['abstractText'] = data_drop['abstractText'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    # Suppression des ponctuations
    data_drop['abstractText'] = data_drop['abstractText'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

    return data_drop

# Fonction pour séparer les données en train et test
def split_data(data):
    X = data['abstractText']
    y = data.drop(columns=['abstractText'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# Fonction pour vectoriser les textes avec TF-IDF
def tfidf_vectorizer(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

# Fonction pour réduire la dimensionnalité avec SVD
def svd(X_train_tfidf, X_test_tfidf, n_components):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    return X_train_svd, X_test_svd, svd

# Fonction pour afficher les mots les plus importants
def print_top_words(model, feature_names, n_top_words): 
    for topic_idx, topic in enumerate(model.components_): 
        message = "Concept #%d: " % topic_idx 
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) 
        print(message) 
    print() 

# Fonction pour entrainer un modèle
def run_models(X_train, y_train, X_test, y_test):
    # Classifieurs de base
    estimators = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    evaluations = []

    # Classifier Chain
    for name, estimator in estimators.items():
        chain = ClassifierChain(estimator)
        chain.fit(X_train, y_train)
        y_pred = chain.predict(X_test)
        evaluation = evaluate_model([(f"Classifier Chain + {name}", y_test, y_pred)])
        evaluations.append(evaluation)
        print(evaluation)

    # Multi Output Classifier
    for name, estimator in estimators.items():
        multi = MultiOutputClassifier(estimator)
        multi.fit(X_train, y_train)
        y_pred = multi.predict(X_test)
        evaluation = evaluate_model([(f"Multi Output Classifier + {name}", y_test, y_pred)])
        evaluations.append(evaluation)
        print(evaluation)
    
    return evaluations

# Fonction pour évaluer un modèle
def evaluate_model(evaluations):
    for name, y_test, y_pred in evaluations:
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        zero_one = zero_one_loss(y_test, y_pred)
        return {
            'model': name,
            'F1 Score (micro)': micro_f1,
            'F1 Score (macro)': macro_f1,
            'Zero-One Loss': zero_one
        }
        
# Fonction pour entrainer un modèle Word2Vec
def entrainement_model(corpus, model_size):
    cores = multiprocessing.cpu_count()
    model = Word2Vec(corpus,vector_size=model_size,sg=0,window=5,min_count=2,workers=cores-1)
    for i in range(100):
        model.train(corpus,total_examples=len(corpus),epochs=1)
        print(i, end=' ')
    model.save('./models/Word2vec_entraine.h5')
    
# Fonction pour vectoriser les textes avec Word2Vec (avec ou sans TF-IDF)
def word2vec_vectoriser(corpus, label, model, tfidf=False, google=False):
    texts = list(corpus)
    if tfidf:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit_transform([' '.join(text) for text in texts])
        df_word2vec = word2vec_tfidf_generator(texts, model, model.vector_size, tfidf_vectorizer, google)
    else:
        df_word2vec = word2vec_generator(texts, model, model.vector_size, google)
    X_train, X_test, y_train, y_test = train_test_split(df_word2vec, label, test_size=0.5, random_state=42)
    df_X_train = pd.DataFrame(X_train)
    df_X_test = pd.DataFrame(X_test)
    run_models(df_X_train, y_train, df_X_test, y_test)

    
# Fonction pour vectoriser les textes avec Word2Vec (sans TF-IDF)
def word2vec_generator(texts, model, vector_size, google=False):
    dict_word2vec = {}
    for index, word_list in enumerate(texts):
        arr = np.zeros(vector_size)  # Initialiser le vecteur moyen
        nb_word = 0
        for word in word_list:
            try:
                if google:
                    arr += model[word]
                else:
                    arr += model.wv[word]
                nb_word += 1
            except KeyError:
                continue
        if nb_word > 0:
            dict_word2vec[index] = arr / nb_word  # Moyenne des embeddings
        else:
            dict_word2vec[index] = arr  # Zéro si aucun mot n'est présent dans le modèle
    df_word2vec = pd.DataFrame(dict_word2vec).T
    return df_word2vec

# Fonction pour vectoriser les textes avec Word2Vec pondéré par TF-IDF
def word2vec_tfidf_generator(texts, model, vector_size, tfidf_vectorizer, google=False):
    tfidf_scores = tfidf_vectorizer.transform([' '.join(text) for text in texts])  # TF-IDF pour chaque phrase
    dict_word2vec = {}
    for index, word_list in enumerate(texts):
        arr = np.zeros(vector_size)  # Initialiser le vecteur moyen pondéré
        total_weight = 0
        for word in word_list:
            try:
                idx = tfidf_vectorizer.vocabulary_.get(word, None)
                if idx is not None:
                    weight = tfidf_scores[index, idx]  # Score TF-IDF du mot
                    if google:
                        arr += model[word] * weight
                    else:
                        arr += model.wv[word] * weight
                    total_weight += weight
            except KeyError:
                continue
        if total_weight > 0:
            dict_word2vec[index] = arr / total_weight  # Moyenne pondérée
        else:
            dict_word2vec[index] = arr  # Zéro si aucun mot n'a un score TF-IDF
    df_word2vec = pd.DataFrame(dict_word2vec).T
    return df_word2vec

# Fonction pour évaluer le meilleur modèle
def best_model(model, data):
    data = prepare_data(data)
    corpus = data['abstractText']
    corpus = corpus.apply(lambda line : gensim.utils.simple_preprocess((line)))
    corpus = list(corpus)
    df_word2vec = word2vec_generator(corpus, model, model.vector_size)

    X_train, X_test, y_train, y_test = train_test_split(df_word2vec, data.drop(columns=['abstractText']), test_size=0.5, random_state=42)
        
    pipeline = Pipeline([
        ('Multi Output Classifier + Random Forest', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    evaluation = evaluate_model([("Multi Output Classifier + Random Forest", y_test, y_pred)])
    print(evaluation)
    return evaluation