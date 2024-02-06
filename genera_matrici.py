from PIL import Image, ImageDraw, ImageFont
import random

# Impostazioni per l'immagine
dimensione_immagine = (300, 300)
sfondo = "white"
colore_linea = "black"
spessore_linea = 2
dimensione_griglia = 8
cell_size = dimensione_immagine[0] // dimensione_griglia

# Impostazioni per il testo
font_size = 20
font = ImageFont.truetype("arial.ttf", font_size)

# Creazione dell'immagine
immagine = Image.new("RGB", dimensione_immagine, sfondo)
disegno = ImageDraw.Draw(immagine)

# Funzione per disegnare la griglia
def disegna_griglia(disegno, dimensione_immagine, dimensione_griglia, spessore_linea):
    for i in range(0, dimensione_immagine[0]+1, cell_size):
        # Linee verticali
        disegno.line([(i, 0), (i, dimensione_immagine[1])], fill=colore_linea, width=spessore_linea)
        # Linee orizzontali
        disegno.line([(0, i), (dimensione_immagine[0], i)], fill=colore_linea, width=spessore_linea)

# Funzione per generare il contenuto delle celle
def genera_contenuto_celle():
    simboli = ['X', 'S', 'T','0','1','2','3','4'] + [str(n) for n in range(5)]
    return [[random.choice(simboli) for _ in range(dimensione_griglia)] for _ in range(dimensione_griglia)]

# Funzione per disegnare il contenuto delle celle
def disegna_contenuto(disegno, contenuto, font):
    for y, riga in enumerate(contenuto):
        for x, cella in enumerate(riga):
            w, h = disegno.textsize(cella, font=font)
            disegno.text(
                (x * cell_size + (cell_size - w) / 2, y * cell_size + (cell_size - h) / 2),
                cella,
                font=font,
                fill=colore_linea
            )

# Disegna la griglia
disegna_griglia(disegno, dimensione_immagine, dimensione_griglia, spessore_linea)

# Genera il contenuto delle celle
contenuto = genera_contenuto_celle()

# Disegna il contenuto
disegna_contenuto(disegno, contenuto, font)

# Salva l'immagine
immagine.save("griglia.png")
