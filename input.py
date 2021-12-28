import emotion_mod2 as e
import nltk
from nltk.tokenize import word_tokenize

print("Enter text:")
for i in range(12):
    text = input()
    print(e.emotion(text))
