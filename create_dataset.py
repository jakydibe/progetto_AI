import csv
from PIL import Image
import numpy as np

def processa_immagine(input_image_path):
    # Caricare l'immagine
    with Image.open(input_image_path) as img:
        # Convertire l'immagine in scala di grigi
        img_gray = img.convert("L")
        
        # Convertire l'immagine in un array numpy e poi in un array monodimensionale
        return np.array(img_gray).flatten()

def immagini_a_csv(input_image_paths, output_csv_path):
    # Aprire il file CSV per la scrittura
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Iterare su ogni percorso di immagine
        for path in input_image_paths:
            # Processare l'immagine e ottenere l'array monodimensionale
            img_array = processa_immagine(path)
            
            # Inserire una colonna vuota all'inizio e scrivere l'array monodimensionale in una riga
            csvwriter.writerow([""] + img_array.tolist())

# Utilizzo della funzione
input_image_paths = []
dimensione_matrici = [7,5,4,2]
for im in range(len(dimensione_matrici)):
    for i in range(dimensione_matrici[im]):
        for j in range(dimensione_matrici[im]):
            input_image_paths.append(f'C:\\Users\\jakyd\\Desktop\\progetto_AI\\immagini\\sotto_immagine_{im}_{i}_{j}.png')
output_csv_path = 'output.csv'
immagini_a_csv(input_image_paths, output_csv_path)
