import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import indian
from nltk.tokenize import PunktSentenceTokenizer


from nltk.stem.mahmud2014 import stem_verb


##from bengali_stemmer.rafikamal2014 import RafiStemmer
##from nltk.stem import BengaliStemmer

text=open('F:\\BACKUP\\A.cuet\\final project\\corpus\\sad.txt', 'r', encoding="utf8").read()
#text="running younger young"
#print(sent_tokenize(text)) #not bengla
#print(word_tokenize(text))
##for i in word_tokenize(text):
##          print(i)
stop_words = set(stopwords.words("bangla"))
words = word_tokenize(text)

#print(stop_words)
filtered_sent = []

for w in words:
   if w not in stop_words:
       filtered_sent.append(w)

#print(filtered_sent)

filtered_sent = [w for w in words if not w in stop_words]
#print(filtered_sent)
##ps = PorterStemmer()
#ps = stem_verb(str)

for w in filtered_sent:
    print(stem_verb(w))


##stemmer = RafiStemmer()
##stemmer.stem_word('বাংলায়')


train_text = indian.raw("bangla.pos")
sample_text = open('F:\\BACKUP\\A.cuet\\final project\\corpus\\anger.txt', 'r', encoding="utf8").read()
##for line in file:
##   sample_text=line.split()
##sample_text = indian.raw("anger.text")
##print(sample_text)

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
   try:
      for i in tokenized:
         words = nltk.word_tokenize(i)
         tagged = nltk.pos_tag(words)
         print(tagged)

         chunckGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """
         chunkParser = nltk.RegexpParser(chunckGram)
         chunked = chunkParser.parse(tagged)

         #chunked.draw()

   except exception as e:
      print(str(e))

process_content()



##sklearn.model_selection.train_test_split

#documents=[(list(movie_reviews.words(fileid)), category)
 #          for category in movie_reviews.categories()
  #         for fileid in movie_reviews.fileids(category)]

##documents = []
##
##for category in movie_reviews.categories():
##   for fileid in movie_reviews.fileids(category):
##      documents.append(list(movie_reviews.words(fileid)), category)

#random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
   all_words.append(w.lower())
