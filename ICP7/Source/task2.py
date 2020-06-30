from bs4 import BeautifulSoup
import requests
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk import trigrams
from nltk import ne_chunk

URL = 'https://en.wikipedia.org/wiki/Google'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
list = []
file = open("input.txt", "w", encoding='utf-8')
file.write(soup.body.text.encode('utf-8',"rb").decode('utf-8'))
file.close()

text = open('input.txt', encoding="utf8").read()
print(text)
#Tokenisation
print("Tokenization")
tokenization = nltk.word_tokenize(text)
tokenization1 = nltk.sent_tokenize(text)
print(tokenization)
print(tokenization1)
#POS
print("Parts of Speech")
POS = nltk.pos_tag(tokenization)
print(POS)
#Stemming

print(" Porter STEMMING:")
porterStemmer = PorterStemmer()
print(porterStemmer.stem(text))

print(" Lancaster STEMMING:")
lancasterStemmer = LancasterStemmer()
print(lancasterStemmer.stem(text))

print(" Snowball STEMMING:")
snowballStemmer = SnowballStemmer('english')
print(snowballStemmer.stem(text))

#Lemmatization
print("Lemmatizing")
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize(text))

print("TRIGRAMS:")
trig = trigrams(tokenization)
for x in trig:
  print(x)
#Named Entity Recognition
print("Named Entity Recognition")
ner = ne_chunk(POS)
print("\nNamed Entity Recognition :", ner)