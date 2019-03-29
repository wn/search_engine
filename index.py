from __future__ import print_function
from collections import defaultdict, Counter
import cPickle
import getopt
from math import sqrt, log
from os import listdir
from os.path import join
import sys

import nltk
from nltk.stem.porter import PorterStemmer

from data_structures import LinkedList


def usage():
    """
    Prints the usage message.
    """
    print("usage: " + sys.argv[0] +
          " -i directory-of-documents -d dictionary-file -p postings-file")


def build_index(input_directory):
    """
    Builds an inverted index out of the files in the input_directory.
    """
    filenames = sorted(read_directory(input_directory))
    index = defaultdict(LinkedList)
    doc_vector_lengths = {}
    all_docs_length = len(filenames)
    for doc_id, file_path in filenames:
        token_count = get_token_weights(file_path)
        for token, count in token_count.items():
            index[token].append((doc_id, count))
        doc_vector_lengths[doc_id] = get_document_vector_length(token_count)

    return index, doc_vector_lengths, all_docs_length


def get_document_vector_length(token_count):
    """
    Calculates the vector normalisation factor.
    """
    return sqrt(sum(val**2 for val in token_count.values()))


def read_directory(input_directory):
    """
    Returns a sorted list of filenames in input_directory, assuming all
    the filenames are integers.
    """
    files = listdir(input_directory)
    complete_filenames = [(int(filename), join(input_directory, filename))
                          for filename in files]
    return complete_filenames


def get_token_weights(filename):
    """
    Tokenise the text contained in the given filename.
    """
    token_count = Counter()
    with open(filename, "r") as filename_file:
        for sentence in nltk.sent_tokenize(filename_file.read()):
            for word in nltk.word_tokenize(sentence):
                token_count[normalise(word)] += 1
    for key in token_count.keys():
        token_count[key] = get_weighted_tf(token_count[key])
    return token_count


def normalise(word, cache={}):
    """
    Normalises the word using stemmer from NLTK.
    """
    word = word.lower()
    if word in cache:
        return cache[word]
    result = PorterStemmer().stem(word)
    cache[word] = result
    return result


def get_idf(all_docs_length, val):
    """
    Calculates the inverse document frequency.
    """
    return log((float(all_docs_length) / val), 10)


def get_weighted_tf(count):
    """
    Calculates the weighted term frequency.
    """
    return 1 + log(count, 10)


def store_index_vector_lengths(index, vector_lengths, output_file_dictionary,
                               output_file_postings, num_documents):
    """
    Stores the index into the given dictionary file and postings file.
    """
    dictionary = {}
    offset = 0
    with open(output_file_postings, "wb") as postings_file:
        for key, postings in index.items():
            pickled = cPickle.dumps(postings, 2)
            postings_file.write(pickled)
            length = len(pickled)
            dictionary[key] = (get_idf(num_documents, len(postings)), offset,
                               length)
            offset += length
    # Stores vector lengths as a nested dictionary in the dictionary
    dictionary["LENGTHS"] = vector_lengths
    with open(output_file_dictionary, "wb") as dictionary_file:
        cPickle.dump(dictionary, dictionary_file, 2)


def main():
    """
    The main function of this file.
    """
    input_directory = output_file_dictionary = output_file_postings = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':  # input directory
            input_directory = arg
        elif opt == '-d':  # dictionary file
            output_file_dictionary = arg
        elif opt == '-p':  # postings file
            output_file_postings = arg
        else:
            assert False, "unhandled option"

    if any(x is None for x in
           [input_directory, output_file_postings, output_file_dictionary]):
        usage()
        sys.exit(2)
    print("Building index")
    index, vector_lengths, num_documents = build_index(input_directory)
    print("Storing index")
    store_index_vector_lengths(index, vector_lengths, output_file_dictionary,
                               output_file_postings, num_documents)


if __name__ == "__main__":
    main()
