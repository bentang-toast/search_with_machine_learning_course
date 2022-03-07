import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import gensim
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

def transform_name(product_name):
    # IMPLEMENT
    product_name = clean_text(product_name)
    #print ("product_name: {}".format(product_name))
    return product_name



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
    text = NUMERIC_RE.sub('', text) # delete NUMERIC character
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stop worlds from text
    #print("pre-stem text: {}".format(text))
    words = word_tokenize(text)
    #print("words: {}".format(words))
    #text = ' '.join(ps.stem(word) for word in words) # use PorterStemmer
    text = ' '.join(stemmer.stem(word) for word in words) # use SnowballStemmer
    #print("text: {}".format(text))
    return text



# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
print(min_products)
sample_rate = args.sample_rate

df = pd.DataFrame()
df_prod_count = pd.DataFrame()
df_cat_prod = pd.DataFrame()

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            
            #df_cat = pd.read_xml(f, xpath='//product/categoryPath/category[last()]')
            #df_prod = pd.read_xml(f, xpath='//product/name[last()]')
            #print(df_cat.shape)
            #print(df_prod.head)

            #df_prod_count = df['id'].value_counts()
            #print(df_prod_count[0])
            
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId
                      cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                      
                      to_append = [cat, name]
                      a_series = pd.Series(to_append)
                      df_cat_prod = df_cat_prod.append(a_series, ignore_index=True)

                      #output.write("__label__%s %s\n" % (cat, transform_name(name)))

    # product count for each category
    print(df_cat_prod.head())
    print(df_cat_prod.shape)
    df_cat_prod.columns =['cat', 'prod']

    if min_products > 0:
        df_cat_prod = df_cat_prod.groupby('cat').filter(lambda x : len(x)>min_products)
        print(df_cat_prod.head())
        print(df_cat_prod.shape)
        for index, row in df_cat_prod.iterrows():
            #print(row['cat'], row['prod'])
            output.write("__label__%s %s\n" % (row['cat'], transform_name(row['prod'])))
    




