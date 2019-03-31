#!/usr/bin/python

from __future__ import print_function
import cPickle
import getopt
import sys
import heapq
from collections import Counter
from math import log
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize

from data_structures import LinkedList


def usage():
    """
    Prints the usage message.
    """
    print("usage: " + sys.argv[0] +
          " -d dictionary-file -p postings-file -q file-of-queries" +
          " -o output-file-of-results")


#######################
# Parsing and loading #
#######################


def parse(query):
    """
    Parses a query into a dictionary where key is the token and its value is
    the term frequency
    """
    tokens = (normalise(token) for token in word_tokenize(query))
    return Counter(tokens)


def get_weighted_tf(count):
    """
    Calculates the weighted term frequency
    using the 'logarithm' scheme.
    """
    BASE = 10
    return log(BASE * count, BASE)


def get_weighted_tfs(counts):
    """
    Calculate the weighted term frequencies.
    """
    return dict(
        map(
            lambda key, val: (key, get_weighted_tf(val),
            counts.items())))


def normalise(token, cache={}):
    """
    Returns a normalised token. Normalised tokens are cached for performance
    """
    token = token.lower()
    if token in cache:
        return cache[token]
    result = PorterStemmer().stem(token)
    cache[token] = result
    return result


def load_postings(postings_file, dictionary, term):
    """
    Loads postings from postings file using memory
    location provided by dictionary.
    """
    # Returns an empty linkedlist if term is not in dictionary
    if term not in dictionary:
        return LinkedList()
    _, offset, length = dictionary[term]
    postings_file.seek(offset)
    pickled = postings_file.read(length)
    return cPickle.loads(pickled)


def load_dictionary(dictionary_file_location):
    """
    Loads dictionary from dictionary file location
    """
    with open(dictionary_file_location, 'r') as dictionary_file:
        dictionary, bitriword_dictionary = cPickle.load(dictionary_file)
    return dictionary


####################
# Query processing #
####################
def process_queries(dictionary, postings_file_location,
                    file_of_queries_location, file_of_output_location):
    """
    Process all the queries in the queries file.
    """
    with open(file_of_queries_location, 'r') as queries, \
            open(postings_file_location, 'rb') as postings_file, \
            open(file_of_output_location, 'w') as output_file:
        for query in queries:
            try:
                process_query(query, dictionary, postings_file, output_file)
            except Exception as e:
                output_file.write("\n")


def process_query(query, dictionary, postings_file, output_file):
    """
    Calculates the cosine scores of the documents, get 10 documents with the
    highest scores and writes to file
    """
    scores = Counter()
    lengths = dictionary["LENGTHS"]
    weighted_tfs = get_weighted_tfs(parse(query))
    for term, tf_q in weighted_tfs.items():
        if term not in dictionary:
            continue
        postings = load_postings(postings_file, dictionary, term)
        idf = dictionary[term][0]
        for doc, tf_d in postings:
            scores[doc] += tf_d * tf_q * idf
    for doc in scores.keys():
        scores[doc] = float(scores[doc]) / lengths[doc]

    postings = " ".join(str(x) for x in retrieve_top_ten_scores(scores))
    output_file.write(postings + "\n")


def retrieve_top_ten_scores(scores):
    """
    Retrieve the top 10 postings by score. This is done using the heapq.
    The results are then sorted by decreasing score and then by increasing
    lexicographical order of terms.
    """
    score_list = heapq.nlargest(10, scores.items(), lambda x: (x[1], -x[0]))
    return (x[0] for x in score_list)


def main():
    """
    The main function of this file.
    """
    dictionary_file = postings_file = file_of_queries = file_of_output = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-d':
            dictionary_file = arg
        elif opt == '-p':
            postings_file = arg
        elif opt == '-q':
            file_of_queries = arg
        elif opt == '-o':
            file_of_output = arg
        else:
            assert False, "unhandled option"

    if any(x is None for x in
           [dictionary_file, postings_file, file_of_queries, file_of_output]):
        usage()
        sys.exit(2)

    dictionary = load_dictionary(dictionary_file)

    process_queries(dictionary, postings_file, file_of_queries, file_of_output)


if __name__ == "__main__":
    main()
