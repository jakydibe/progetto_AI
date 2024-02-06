import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


num_images = 5000


# Carica i file CSV del dataset EMNIST (assicurati di specificare il percorso corretto)
print("prima di leggere i file")


train_data = pd.read_csv('C:\\Users\\jakyd\\Desktop\\progetto_AI\\dataset\\emnist-mnist-train.csv', header=None)
test_data = pd.read_csv('C:\\Users\\jakyd\\Desktop\\progetto_AI\\dataset\\emnist-mnist-test.csv', header=None)

#train_data = pd.concat([train_data_letters[:num_images], train_data_numbers[:num_images]], ignore_index=True)
#test_data = pd.concat([test_data_letters[num_images:num_images+1000], test_data_numbers[num_images:num_images+1000]], ignore_index=True)

# Mescola i dati di addestramento (train_data)
train_data = shuffle(train_data, random_state=42)

# Mescola i dati di test (test_data)
test_data = shuffle(test_data, random_state=42)


train_data = train_data.iloc[:num_images, :]

print("prima di dividere dati in feature e target") 

# Divide i dati in feature (X) e target (y)
X_train = train_data.iloc[:, 1:].values.astype('float32') / 255.0  # Normalizza le feature
y_train = train_data.iloc[:, 0].values.astype('int')

X_test = test_data.iloc[:, 1:].values.astype('float32') / 255.0  # Normalizza le feature
y_test = test_data.iloc[:, 0].values.astype('int')


print("prima di creare il classificatore")

# Crea un classificatore MLP (Multi-layer Perceptron), utilizza la sigmoide come funzione di attivazione
classifier = MLPClassifier(hidden_layer_sizes=(256,128, 36),solver='adam', max_iter=500,alpha=0.001, activation='relu', random_state=88)

print("prima di addestrare")

# Addestra il classificatore
classifier.fit(X_train, y_train)

print("prima di valutare")


# Valuta il modello
y_pred = classifier.predict(X_test[:1000])
accuracy = accuracy_score(y_test[:1000], y_pred)
print(f'Accuracy: {accuracy:.2f}')
