import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

import re
from bs4 import BeautifulSoup

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
  
categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'
cleaned_query_file_name = r'/workspace/datasets/cleaned_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")
general.add_argument("--skip_clean", default=None, help="skip normalize queries")
general.add_argument("--cleaned_query_file", default=cleaned_query_file_name, help="the cleaned query file name")


args = parser.parse_args()
output_file_name = args.output
cleaned_query_file_name = args.cleaned_query_file

if args.min_queries:
    min_queries = int(args.min_queries)

nltk.download('stopwords')
nltk.download('punkt')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
PUNCTIONIONS_RE = re.compile('[^\w\s]')
NUMERIC_RE = re.compile(" \d+")
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    #stemmer = SnowballStemmer("english")
    #text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() #lowercase text
    #text = text.replace('\n', ' ')  # replace new line with space
    text = PUNCTIONIONS_RE.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in a BAD_SYMBOLS_RE from text
    #text = NUMERIC_RE.sub('', text) # delete NUMERIC character
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stop worlds from text
    words = word_tokenize(text)
    text = ' '.join(stemmer.stem(word) for word in words) # use SnowballStemmer
    return text


# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

#print(parents_df['category'].nunique())

if (args.skip_clean):
    df = pd.read_csv(cleaned_query_file_name)[['category', 'query']]
else:
    # Read the training data into pandas, only keeping queries with non-root categories in our category tree.
    df = pd.read_csv(queries_file_name)[['category', 'query']]
    df = df[df['category'].isin(categories)]

    # IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
    print (df)
    df['query'] = df['query'].map(lambda x: clean_text(x))
    #df['query'] = df['query'].str.lower()
    #df['query'] = df['query'].str.replace('\n', ' ') 
    #df['query'] = df['query'].map(lambda x: x.strip('"'))
    #df['query'] = df['query'].map(lambda x: PUNCTIONIONS_RE.sub('', x))
    #df['query'] = df['query'].map(lambda x: REPLACE_BY_SPACE_RE.sub(' ', x))
    #df['query'] = df['query'].map(lambda x: BAD_SYMBOLS_RE.sub(' ', x))
    #df['query'] = df['query'].map(lambda x: NUMERIC_RE.sub(' ', x))
    #df['query'] = df['query'].map(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
    #print (df)
    print (df.nunique())
    
    # write to CSV after queries cleaned
    df.to_csv(cleaned_query_file_name)

print (df)
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
# count all categories then do a join
df_count = df.groupby('category').size().reset_index(name='count')
df_merged = df.merge(df_count, how='left', on='category').merge(parents_df, how='left', on='category')
df = df_merged


# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
