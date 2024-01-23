import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time
from PIL import Image

num_images = 1000


# Carica i file CSV del dataset EMNIST (assicurati di specificare il percorso corretto)
print("prima di leggere i file")

train_data_letters = pd.read_csv('C:\\Users\\jakyd\\Desktop\\progetto_AI\\dataset\\emnist-letters-train.csv', header=None)
test_data_letters = pd.read_csv('C:\\Users\\jakyd\\Desktop\\progetto_AI\\dataset\\emnist-letters-test.csv', header=None)

print(train_data_letters.shape)
print(test_data_letters.shape)

#togliere tutto cio' che non e' X=24, S=19, T=20 + 10 -> 34, 29, 30


print("pulendo le lettere")

filtro_train = (train_data_letters.iloc[:, 0] == 24) | (train_data_letters.iloc[:, 0] == 19) | (train_data_letters.iloc[:, 0] == 20)
filtro_test = (test_data_letters.iloc[:, 0] == 24) | (test_data_letters.iloc[:, 0] == 19) | (test_data_letters.iloc[:, 0] == 20)



train_data_letters = train_data_letters[filtro_train]
test_data_letters = test_data_letters[filtro_test]



for y in range(len(train_data_letters.iloc[:, 0])):
    train_data_letters.iloc[y, 0] += 10


for y in range(len(test_data_letters.iloc[:, 0])):
    test_data_letters.iloc[y, 0] += 10



train_data_numbers = pd.read_csv('C:\\Users\\jakyd\\Desktop\\progetto_AI\\dataset\\emnist-mnist-train.csv', header=None)
test_data_numbers = pd.read_csv('C:\\Users\\jakyd\\Desktop\\progetto_AI\\dataset\\emnist-mnist-test.csv', header=None)

#togliere 0,5,6,7,8,9, corrispondono letteralmente
print("pulendo i numeri")

filtro_train_numbers = (train_data_numbers.iloc[:, 0] <= 4) & (train_data_numbers.iloc[:, 0] != 0)
filtro_test_numbers = (test_data_numbers.iloc[:, 0] <=4) & (test_data_numbers.iloc[:, 0] != 0)


train_data_numbers = train_data_numbers[filtro_train_numbers]
test_data_numbers = test_data_numbers[filtro_test_numbers]  

train_data_numbers = shuffle(train_data_numbers, random_state=42)
test_data_numbers = shuffle(test_data_numbers, random_state=42)
train_data_letters = shuffle(train_data_letters, random_state=42)
test_data_letters = shuffle(test_data_letters, random_state=42)

train_data = pd.concat([train_data_letters[:num_images], train_data_numbers[:num_images]], ignore_index=True)
test_data = pd.concat([test_data_letters[:num_images], test_data_numbers[:num_images]], ignore_index=True)



# Mescola i dati di addestramento (train_data)
train_data = shuffle(train_data, random_state=42)


print(train_data)
# Mescola i dati di test (test_data)

train_data = train_data.iloc[:num_images, :]

print("prima di dividere dati in feature e target") 

# Divide i dati in feature (X) e target (y)
X_train = train_data.iloc[:, 1:].values.astype('float32') / 255.0  # Normalizza le feature
y_train = train_data.iloc[:, 0].values.astype('int')

#x_train_new = []
#y_train_new = []



#accepted_char = ['X', 'S', 'T']

# for s in y_train:
#     print(s)
#     if ((int(s) <= 4) and (int(s) >= 1)) or (s in accepted_char):
#         x_train_new.append(s)
#         y_train_new.append(s)


X_test = test_data.iloc[:, 1:].values.astype('float32') / 255.0  # Normalizza le feature
y_test = test_data.iloc[:, 0].values.astype('int')

# x_test_new = []
# y_test_new = []

print(X_test)

# for s in y_train:
#     if ((int(s) <= 4) and (int(s) >= 1)) or (s in accepted_char):
#         x_test_new.append(s)
#         y_test_new.append(s)


# print(y_train_new)
# print(y_test_new)

print("prima di creare il classificatore")

# Crea un classificatore MLP (Multi-layer Perceptron), utilizza la sigmoide come funzione di attivazione
classifier = MLPClassifier(hidden_layer_sizes=(256,128,36),solver='adam', max_iter=500, activation='logistic')

print("prima di addestrare")

# Addestra il classificatore
classifier.fit(X_train, y_train)

print("prima di valutare")


# Valuta il modello
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


img = Image.open('sotto_immagine_0_3.png')
img_gray = img.convert('L')
#img_gray = img_gray.resize((28, 28), Image.ANTIALIAS)

print(img_gray)
img_array = np.asarray(img_gray)
img_array = img_array / 255.0
img_array = img_array.reshape(1, 784)

img_pred = classifier.predict(img_array)
print(f"PREDICTION IMMAGINE: {img_pred}")