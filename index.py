"""
Creates an index for our SearchEngine
"""

from collections import defaultdict, Counter
import pickle
import getopt
from math import sqrt, log
from functools import lru_cache
import sys
import csv
import time
import gc

import nltk
from nltk.stem.porter import PorterStemmer
from joblib import Parallel, delayed

from data_structures import LinkedList


def usage():
    """
    Prints the usage message.
    """
    print("usage: " + sys.argv[0] +
          " -i directory-of-documents -d dictionary-file -p postings-file")


# def build_bitriword_index(data):
#     """
#     Builds an inverted index out of the files in the input_file.
#     A compound index with both biword indexes and triword tokens is built.
#     """
#     index = defaultdict(LinkedList)
#     print("Generating all biword/triword token sets")
#     all_bitriword_tokens = [
#         get_bitriword_tokens(content) for _, content in data
#     ]
#     index["ALL"].extend(doc_id for doc_id, _ in data)
#     print("Adding the tokens to the index")
#     for (doc_id, _), bitriword_tokens in zip(data, all_bitriword_tokens):
#         for token in bitriword_tokens:
#             # None is the second element appended as no relevant weights
#             index[token].append(doc_id)
#     print("Building skips...")
#     for postings in index.values():
#         postings.build_skips()
#     return index

# def get_bitriword_tokens(content):
#     """
#     Tokenise the text contained in the given filename to biword
#     and triword tokens.
#     """
#     # Build biword
#     biword_tokens = {
#         " ".join(content[i:i + 2])
#         for i in range(len(content) - 1)
#     }
#     # Build triword
#     triword_tokens = {
#         " ".join(content[i:i + 3])
#         for i in range(len(content) - 2)
#     }
#     return biword_tokens.union(triword_tokens)


def build_positional_index(data):
    """
    Builds a positional index out of the rows in the data.
    """
    index = defaultdict(LinkedList)
    for doc_id, content in data:
        positions_index = defaultdict(LinkedList)
        for i, token in enumerate(content):
            positions_index[token].append(i)
        for token, positions in positions_index.items():
            index[token].append((doc_id, positions))
    return index


def build_tfidf_index(data):
    """
    Builds both a tf-idf index from the data.
    """
    index = defaultdict(LinkedList)
    doc_vector_lengths = {}
    all_docs_length = len(data)
    all_token_count = [get_token_weights(content) for _, content in data]
    doc_vector_lengths = {
        doc_id: get_document_vector_length(token_count)
        for (doc_id, _), token_count in zip(data, all_token_count)
    }
    for (doc_id, _), token_count in zip(data, all_token_count):
        for token, count in token_count.items():
            index[token].append((doc_id, count))

    return index, doc_vector_lengths, all_docs_length


def get_token_weights(content):
    """
    Tokenise the text contained in the given filename.
    """
    token_count = Counter(content)
    return {k: get_weighted_tf(v) for k, v in token_count.items()}


def get_document_vector_length(token_count):
    """
    Calculates the vector normalisation factor
    using the 'cosine normalization' scheme.
    """
    return sqrt(sum(val**2 for val in token_count.values()))


def read_data_file(input_file):
    """
    Return a list of data sorted by the file name
    """
    csv.field_size_limit(sys.maxsize)
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        return Parallel(
            n_jobs=-1, verbose=10, backend="multiprocessing")(
                delayed(parse_row)(row) for row in reader)


def parse_row(row):
    """
    Parses the content by tokenising the content and normalising each word
    """
    return (row[0], [normalise(word) for word in nltk.word_tokenize(row[2])])


@lru_cache(maxsize=None)
def normalise(word):
    """
    Normalises the word using stemmer from NLTK.
    """
    word = word.lower()
    return PorterStemmer().stem(word)


def get_idf(all_docs_length, val):
    """
    Calculates the inverse document frequency using
    the 'inverse collection frequency' scheme.
    """
    return log((float(all_docs_length) / val), 10)


def get_weighted_tf(count, base=10):
    """
    Calculates the weighted term frequency using the
    'logarithm' scheme.
    """
    return log(base * count, base)


def store_to_postings_file(index, positional_index, output_file_postings,
                           num_documents):
    """
    Stores the postings in index and the positional index in positional_index
    to postings file and generate a dictionary to access the postings file.
    """
    dictionary = {}
    with open(output_file_postings, "wb") as postings_file:
        tokens = set(index.keys()).union(set(positional_index.keys()))
        for token in tokens:
            postings = index[token]
            postings_offset, postings_length = pickle_to_file(
                postings_file, postings)
            positional_offset, positional_length = pickle_to_file(
                postings_file, positional_index[token])
            dictionary[token] = (get_idf(num_documents, len(postings)),
                                 (postings_offset, postings_length),
                                 (positional_offset, positional_length))
    return dictionary


def store_to_dictionary_file(dictionary, vector_lengths,
                             output_file_dictionary):
    """
    Stores a tuple of dictionary and vector_lengths to the dictionary file.
    """
    with open(output_file_dictionary, "wb") as dictionary_file:
        pickle.dump((dictionary, vector_lengths), dictionary_file,
                    pickle.HIGHEST_PROTOCOL)


def pickle_to_file(postings_file, linked_list):
    """
    Stores the given linked_list object to a file object postings_file
    """
    offset = postings_file.tell()
    pickled = pickle.dumps(linked_list, pickle.HIGHEST_PROTOCOL)
    length = len(pickled)
    postings_file.write(pickled)
    return (offset, length)


def main():
    """
    The main function of this file.
    """
    input_file = output_file_dictionary = output_file_postings = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':  # input directory
            input_file = arg
        elif opt == '-d':  # dictionary file
            output_file_dictionary = arg
        elif opt == '-p':  # postings file
            output_file_postings = arg
        else:
            assert False, "unhandled option"

    if any(x is None for x in
           [input_file, output_file_postings, output_file_dictionary]):
        usage()
        sys.exit(2)
    start_time = cur_time = time.time()
    print("Building index")

    print("1. Retrieving data")
    data = read_data_file(input_file)
    print("Time taken = " + str(time.time() - cur_time))

    print("2. Sorting data")
    cur_time = time.time()
    data = sorted(data)
    print("Time taken = " + str(time.time() - cur_time))

    print("3. Building tf-idf index")
    cur_time = time.time()
    index, vector_lengths, num_documents = build_tfidf_index(data)
    print("Time taken = " + str(time.time() - cur_time))

    # print("4. Building Biword Triword index")
    # cur_time = time.time()
    # bitriword_index = build_bitriword_index(data)
    # print("Time taken = " + str(time.time() - cur_time))

    print("4. Building positional index")
    cur_time = time.time()
    positional_index = build_positional_index(data)
    print("Time taken = " + str(time.time() - cur_time))

    print("5. Storing to postings file")
    cur_time = time.time()
    dictionary = store_to_postings_file(index, positional_index,
                                        output_file_postings, num_documents)
    print("Time taken = " + str(time.time() - cur_time))

    print("6. Storing to dictionary file")
    cur_time = time.time()
    store_to_dictionary_file(dictionary, vector_lengths,
                             output_file_dictionary)
    print("Time taken = " + str(time.time() - cur_time))

    print("Total time = " + str(time.time() - start_time))


if __name__ == "__main__":
    # In Python, memory management is mainly through refcount, GC is only
    # used for cycle detection. As long as our data structures do not contain
    # cycles, it is safe to disable the GC.
    gc.disable()
    main()
