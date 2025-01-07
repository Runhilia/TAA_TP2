from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

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
        evaluation = {name: evaluate_model("Classifier Chain + " + estimator[0], y_test, y_pred)}
        evaluations.append(evaluation)
        print(evaluation)


    # Multi Output Classifier
    for name, estimator in estimators.items():
        multi = MultiOutputClassifier(estimator)
        multi.fit(X_train, y_train)
        y_pred = multi.predict(X_test)
        evaluation = {name: evaluate_model("Multi Output Classifier + " + estimator[0], y_test, y_pred)}
        evaluations.append(evaluation)
        print(evaluation)
    
    return evaluations

def evaluate_model(evaluations):
    for name, y_test, y_pred in evaluations:
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        zero_one = zero_one_loss(y_test, y_pred)
        return {
            'F1 Score (micro)': micro_f1,
            'F1 Score (macro)': macro_f1,
            'Zero-One Loss': zero_one
        }

