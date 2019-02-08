import string
import math
import spacy
import re
import numpy as np
from collections import OrderedDict
from nltk.corpus import stopwords

def primera_limpieza(texto):
    simb = re.compile(r'[!?¿¡/\\]')
    texto = texto.lower()
    texto = re.sub(simb, '', texto)
    texto = ' '.join(texto.split())
    debris = set(string.printable + 'ÁÉÍÓÚÄËÏÖÜÀÈÌÒÙáéíóúäëïöüàèìòùçÇ')    
    
    # Eliminar los caracteres extraños
    texto = filter(lambda x: x in debris, texto)
    texto = filter(lambda x: x != "\n", texto)
    
    # Eliminar la puntuación da resultados rarunos, evitar.
    # texto = filter(lambda x: x not in string.punctuation, texto)

    return ''.join(texto)

def preprocesamiento(texto, lang):
    
    lang_dic = {
        'es':'spanish',
        'en':'english',
        'fr':'french'}
    
    nlp = spacy.load(lang)
    stop_words = stopwords.words(lang_dic[lang])
    texto_limpio = str(primera_limpieza(texto))
    doc = nlp(texto_limpio)

    processed_text = []  # Lemas con POS y filtrado
    lemmatized_text = []  # Lemas sin filtrado
    wanted_pos = ["VERB","ADJ", "NOUN"]
    unwanted_pos = ["PUNCT", "SYM", "X", "SPACE", "NUM"]
    
    # Generación de stoplist
    for token in doc:
        if token.is_stop:
            stop_words.append(str(token))
        elif token.pos_ in unwanted_pos:
            stop_words.append(str(token))
    
    # Generación de lista de texto procesado        
    for token in doc:
        if token.pos_ in wanted_pos:
            lemmatized_text.append(str(token.lemma_))
        else:
            lemmatized_text.append(str(token))
            stop_words.append(str(token))
    
    stop_words = list(set(stop_words))
    
    for token in lemmatized_text:
        if token not in stop_words:
            processed_text.append(token)
    
    vocabulario = list(set(processed_text))

    return processed_text, lemmatized_text, vocabulario, stop_words


def simpletextrank(processed_text, lemmatized_text, vocabulario, stop_words,
             MAX_ITERATIONS=50, UMBRAL=0.0001, d=0.85, ventana=5, ng=3, n_kw=10):
    
    len_vocabulario = len(vocabulario)
    peso_borde = np.zeros((len_vocabulario, len_vocabulario), dtype=np.float32)
    score = np.zeros((len_vocabulario), dtype=np.float32)
    tamanyo_ventana = ventana
    ocurrencias_cubiertas = []
    in_out = np.zeros((len_vocabulario), dtype=np.float32)
    score_frases = []
    frases_unicas = []
    frases = []
    frase = ' '
    palabras_clave = []
    dict_resultados = []

    # Generación del grafo de entrada con el texto
    for i in range(0, len_vocabulario):
        score[i] = 1
        for j in range(0, len_vocabulario):
            if j == i:
                peso_borde[i][j] = 0
            else:
                for inicio_ventana in range(0, (len(processed_text) - tamanyo_ventana)):
                    final_ventana = inicio_ventana + tamanyo_ventana
                    ventana = processed_text[inicio_ventana:final_ventana]
                    if (vocabulario[i] in ventana) and (vocabulario[j] in ventana):
                        indice_i = inicio_ventana + \
                            ventana.index(vocabulario[i])
                        indice_j = inicio_ventana + \
                            ventana.index(vocabulario[j])
                        if [indice_i, indice_j] not in ocurrencias_cubiertas:
                            peso_borde[i][j] += 1 / \
                                math.fabs(indice_i - indice_j)
                            ocurrencias_cubiertas.append([indice_i, indice_j])

    # Cálculo de la suma con los pesos de la conexión entre vértices
    for i in range(0, len_vocabulario):
        for j in range(0, len_vocabulario):
            in_out[i] += peso_borde[i][j]

    # Puntaje de los vértices
    for iter in range(0, MAX_ITERATIONS):
        score_anterior = np.copy(score)

        for i in range(0, len_vocabulario):
            counter = 0
            for j in range(0, len_vocabulario):
                if peso_borde[i][j] != 0:
                    counter += (peso_borde[i][j] / in_out[j]) * score[j]
            score[i] = (1 - d) + d * (counter)
        if np.sum(np.fabs(score_anterior - score)) <= UMBRAL:  # condición de convergencia
            print("Convergencia en iteración: {}\n".format(str(iter)))
            break

    # Particionado de frases
    for palabra in lemmatized_text:
        if palabra in stop_words:
            if frase != ' ':
                frases.append(str(frase).strip().split())
            frase = ' '
        elif palabra not in stop_words:  # cambiar por otra cosa!
            frase += str(palabra)
            frase += ' '

    # Generación de lista de frases únicas
    for frase in frases:
        if frase not in frases_unicas:
            frases_unicas.append(frase)
    
    # Depuración de lista de palabras        
    for palabra in vocabulario:
        for frase in frases_unicas:
            if (palabra in frase) and ([palabra] in frases_unicas) and (len(frase)>1):
                frases_unicas.remove([palabra])

    # Puntajes de candidatos a palabras clave
    for frase in frases_unicas:
        if len(str(frase).split()) <= ng:
            score_frase = 0  # score por frase != lista de scores
            palabra_clave = ''
            for palabra in frase:
                palabra_clave += str(palabra)
                palabra_clave += ' '
                score_frase += score[vocabulario.index(palabra)]
            score_frases.append(score_frase)
            palabras_clave.append(palabra_clave.strip())

    # print(palabras_clave)
    for pc in palabras_clave:
        counter = palabras_clave.index(pc)
        dict_resultados.append([pc,score_frases[counter]])
    
    dict_resultados.sort(key=lambda v: v[1], reverse=True)

    return dict_resultados[:n_kw]
    
'''
with open('textfile.txt') as f:
    texto = f.read()
    pc, lt, v, sw = preprocesamiento(texto, 'en')
    keywords = simpletextrank(pc, lt, v, sw, ventana=5, ng=3, n_kw=10)
    for i in keywords:
        print('{}\t{}'.format(i[0],i[1]))
'''
