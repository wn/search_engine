from typing import Tuple, List, Dict, BinaryIO
from data_structures import *
from collections import Counter, defaultdict
from search_helpers import normalise, load_postings_list

ALPHA = 1.0
BETA = 0.75


def get_relevant_docs(
        query: str,
        dictionary: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]],
        vector_lengths: Dict[str, float],
        relevant_doc_ids: List[str],
        postings_file: BinaryIO) -> LinkedList:
    query_vector = query_to_vector(query)
    ################################
    # APPLY ROCCIO TO query vector #
    ################################

    scores = defaultdict(float)
    for term, count in query_vector.items():
        if term not in dictionary:
            continue
        term_postings = load_postings_list(postings_file, dictionary, term)
        idf = dictionary[term][0]
        for doc_id, tf_d in term_postings:
            scores[doc_id] += tf_d * idf
    normalized_scores = sorted(
        [(doc_id, score / vector_lengths[doc_id]) for doc_id, score in scores.items()],
        key=lambda x: x[1],
        reverse=True)
    output = LinkedList()
    output.extend(normalized_scores)
    return output


def query_to_vector(query: str) -> Counter:
    return Counter(map(normalise, query.split(' ')))


def rocchio_algorithm(
        query: Dict[str, int],
        relevant_docs: List[str],
        alpha: int,
        beta: int) -> Dict[str, int]:
    normalized_query = {k: alpha * v for k, v in query.items()}
    relevant_docs_centroid = {*filter(lambda x: x in query.keys(),
                                      {k: beta * v / len(relevant_docs) for k, v in sum(relevant_docs.values())})}
    return dict(Counter(normalized_query) + Counter(relevant_docs_centroid))
