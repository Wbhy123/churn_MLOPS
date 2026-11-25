import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# préparation de données
def prepare_data(filename):

    df = pd.read_csv(filename)

    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])

    df = df.drop(['Surname', 'Geography'], axis=1)

    x = df.drop(['Exited', 'RowNumber', 'CustomerId'], axis=1)
    y = df['Exited']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    return x_train, x_test, y_train, y_test


# entrainement modele
def train_model(x_train, y_train):

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(x_train, y_train)

    return rf


# evaluation modele
def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy score:", accuracy * 100)
    print("Confusion Matrix:")
    print(matrix)

    return accuracy, matrix


# enregistrement modele
def save_model(model, filename="classifier.joblib"):

    joblib.dump(model, filename)


# Load model
def load_model(filename="classifier.joblib"):

    model = joblib.load(filename)
    return model
