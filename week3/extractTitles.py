import os
import random
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
  
import re
from bs4 import BeautifulSoup
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
NUMERIC_RE = re.compile(" \d+")
STOPWORDS = set(stopwords.words('english'))
## Tokens to evaluate
#TOKENS = [
#    "phones", "headphones", "laptop", "tablet", "printer", 
#    "sony", "apple", "microsoft", "samsung", "hp",
#    "iphone", "ipad", "galaxy", "windows", "thinkpad",
#    "black", "white", "blue", "32GB", "4G"    
#]

directory = r'/workspace/search_with_machine_learning_course/data/pruned_products'
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing the products")
general.add_argument("--output", default="/workspace/datasets/fasttext/titles.txt", help="the file to output to")

# Consuming all of the product data takes a while. But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=0.1, type=float, help="The rate at which to sample input (default is 0.1)")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input

sample_rate = args.sample_rate

def transform_training_data(name):
    # IMPLEMENT
    name = clean_text(name)
    return name.replace('\n', ' ')



def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    #ps = PorterStemmer()
    stemmer = SnowballStemmer("english")
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() #lowercase text
    text = text.replace('\n', ' ')  # replace new line with space
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in a BAD_SYMBOLS_RE from text
    #text = NUMERIC_RE.sub('', text) # delete NUMERIC character
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stop worlds from text
    #print("pre-stem text: {}".format(text))
    words = word_tokenize(text)
    #print("words: {}".format(words))
    #text = ' '.join(ps.stem(word) for word in words) # use PorterStemmer
    text = ' '.join(stemmer.stem(word) for word in words) # use SnowballStemmer
    #print("text: {}".format(text))
    return text


# Directory for product data

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                if (child.find('name') is not None and child.find('name').text is not None):
                    name = transform_training_data(child.find('name').text)
                    output.write(name + "\n")
