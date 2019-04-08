"""
Processes search queries
"""

import csv
import pickle
import getopt
import sys
from math import log
from functools import lru_cache

from typing import Dict, Tuple, BinaryIO, List, Union

from nltk.stem.porter import PorterStemmer

from .data_structures import LinkedList, TokenType, QueryType



def usage() -> None:
    """
    Prints the usage message.
    """
    print("usage: " + sys.argv[0] +
          " -d dictionary-file -p postings-file -q file-of-queries" +
          " -o output-file-of-results")


#######################
# Parsing and loading #
#######################


def get_weighted_tf(count: int, base: int = 10) -> float:
    """
    Calculates the weighted term frequency
    using the 'logarithm' scheme.
    """
    return log(base * count, base)


def get_weighted_tfs(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate the weighted term frequencies.
    """
    return {k: get_weighted_tf(v) for k, v in counts.items()}


@lru_cache(maxsize=None)
def normalise(token: str) -> str:
    """
    Returns a normalised token. Normalised tokens are cached for performance
    """
    token = token.lower()
    return PorterStemmer().stem(token)


def load_positional_index(
        postings_file: BinaryIO,
        dictionary: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]],
        token: str) -> LinkedList[Tuple[str, LinkedList[int]]]:
    """
    Loads positional index from postings file using the location provided
    by the dictionary.

    Returns an empty LinkedList if token is not in dictionary.
    """
    if token not in dictionary:
        return LinkedList()
    _, _, (offset, length) = dictionary[token]
    postings_file.seek(offset)
    pickled = postings_file.read(length)
    return pickle.loads(pickled)


def load_dictionary(
        dictionary_file_location: str
) -> Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]]:
    """
    Loads dictionary from dictionary file location.
    Returns a tuple of (dictionary, vector_lengths)
    """
    with open(dictionary_file_location, 'rb') as dictionary_file:
        return pickle.load(dictionary_file)


####################
# Query processing #
####################
def process_query(
        dictionary: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]],
        postings_file_location: str, file_of_queries_location: str,
        file_of_output_location: str) -> None:
    """
    Process the query in the query file.
    """
    with open(file_of_queries_location, 'r') as query_file, \
            open(postings_file_location, 'rb') as postings_file, \
            open(file_of_output_location, 'w') as output_file:
        query, *relevant_doc_ids = list(query_file)
        query_type, tokens = parse_query(query)
        if query_type is Query.BOOLEAN:
            return LinkedList()
        elif query_type is Query.FREE_TEXT:
            return LinkedList()



def parse_query(
        query: str) -> Tuple[QueryType, List[Tuple[str, Union[List[str], str]]]]:
    """
    Parses query.

    Clarification from Zhao Jin:
    The queries could be in the form of A B C, A AND B AND C, "A B" AND C,
    but not A B AND C.

    Note that the query is actually a row in a CSV document with
    space delimiter and double quote as the quote character.
    """
    import pdb;pdb.set_trace()
    tokens = list(csv.reader([query], delimiter=' ', quotechar='"'))[0]
    if 'AND' in tokens:
        return (QueryType.BOOLEAN
                [parse_token(token) for token in tokens if token != 'AND'])
    else:
        return (QueryType.FREE_TEXT, [parse_token(token) for token in tokens])


def parse_token(token: str) -> Tuple[TokenType, Union[str, List[str]]]:
    if ' ' in token:
        return (TokenType.PHRASE, [normalise(word) for word in token.split()])
    else:
        return (TokenType.NON_PHRASE, normalise(token))


# def process_query(query, dictionary, postings_file, output_file):
#     """
#     Calculates the cosine scores of the documents, get 10 documents with the
#     highest scores and writes to file
#     """
#    scores = Counter()
#    lengths = dictionary["LENGTHS"]
#    weighted_tfs = get_weighted_tfs(parse(query))
#    for term, tf_q in list(weighted_tfs.items()):
#        if term not in dictionary:
#            continue
#        postings = load_postings(postings_file, dictionary, term)
#        idf = dictionary[term][0]
#        for doc, tf_d in postings:
#            scores[doc] += tf_d * tf_q * idf
#    for doc in list(scores.keys()):
#        scores[doc] = float(scores[doc]) / lengths[doc]
#
#    postings = " ".join(str(x) for x in retrieve_top_ten_scores(scores))
#    output_file.write(postings + "\n")


def main() -> None:
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

    dictionary, vector_lengths = load_dictionary(dictionary_file)

    process_query(dictionary, postings_file, file_of_queries, file_of_output)


if __name__ == "__main__":
    main()
