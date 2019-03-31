from __future__ import print_function
from collections import defaultdict, Counter
import cPickle
import getopt
from math import sqrt, log
import sys
import csv

import nltk
from nltk.stem.porter import PorterStemmer

from data_structures import LinkedList


def usage():
    """
    Prints the usage message.
    """
    print("usage: " + sys.argv[0] +
          " -i directory-of-documents -d dictionary-file -p postings-file")


def build_bitriword_index(data):
    """
    Builds an inverted index out of the files in the input_file.
    A compound index with both biword indexes and triword tokens is built.
    """
    index = defaultdict(LinkedList)
    all_bitriword_tokens = [
        get_bitriword_tokens(content) for _, content in data
    ]
    index["ALL"].extend(doc_id for doc_id, _ in data)
    print("Generated all biword/triword token sets")
    for (doc_id, _), bitriword_tokens in zip(data, bitriword_tokens):
        for token in bitriword_tokens:
            # None is the second element appended as no relevant weights
            index[token].append((doc_id, None))
    for postings in index.values():
        postings.build_skips()
    return index


def get_bitriword_tokens(content):
    """
    Tokenise the text contained in the given filename to biword
    and triword tokens.
    """
    # Build biword
    biword_tokens = {
        " ".join(content[i:i + 2])
        for i in range(len(content) - 1)
    }
    # Build triword
    triword_tokens = {
        " ".join(content[i:i + 3])
        for i in range(len(content) - 2)
    }
    return biword_tokens.union(triword_tokens)


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
    for (doc_id, content), token_count in zip(data, all_token_count):
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
        return [(row[0], parse_content(row[2])) for row in reader]


def parse_content(content):
    content = unicode(content, 'utf-8')
    return [normalise(word) for word in nltk.word_tokenize(content)]


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
    Calculates the inverse document frequency using
    the 'inverse collection frequency' scheme.
    """
    return log((float(all_docs_length) / val), 10)


def get_weighted_tf(count):
    """
    Calculates the weighted term frequency using the
    'logarithm' scheme.
    """
    BASE = 10
    return log(BASE * count, BASE)


def store_indexes(index, vector_lengths, bitriword_indexes,
                  output_file_dictionary, output_file_postings, num_documents):
    """
    Stores the index into the given dictionary file and postings file.
    """
    dictionary = {}
    bitriword_dictionary = {}
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

        for key, value in bitriword_indexes.items():
            pickled = cPickle.dumps(value, 2)
            postings_file.write(pickled)
            length = len(pickled)
            bitriword_dictionary[key] = (len(value), offset, length)
            offset += length

    with open(output_file_dictionary, "wb") as dictionary_file:
        cPickle.dump([dictionary, bitriword_dictionary], dictionary_file, 2)


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
    print("Building index")
    print("1. Retrieving data")
    data = sorted(read_data_file(input_file))
    print("2. Building tf-idf index")
    index, vector_lengths, num_documents = build_tfidf_index(data)
    print("3. Building Biword Triword index")
    bitriword_index = build_bitriword_index(data)
    print("4. Storing index")
    store_indexes(index, vector_lengths, bitriword_index,
                  output_file_dictionary, output_file_postings, num_documents)


if __name__ == "__main__":
    main()
