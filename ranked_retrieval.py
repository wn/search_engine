from typing import Tuple, List, Dict, BinaryIO
from data_structures import LinkedList
from collections import Counter, defaultdict
from search_helpers import normalise, load_postings_list

ALPHA = 1.0
BETA = 0.75
THRESHOLD = 0


def get_relevant_docs(
        query: str,
        dictionary: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]],
        vector_lengths: Dict[str, float],
        relevant_doc_ids: List[str],
        document_vectors_dictionary: Dict[str, Tuple[int, int]],
        postings_file: BinaryIO) -> LinkedList:
    query_vector = query_to_vector(query)
    newQuery = rocchio_algorithm(
                                query_vector,
                                relevant_doc_ids,
                                document_vectors_dictionary,
                                ALPHA,
                                BETA)

    scores = defaultdict(float)
    for term, count in newQuery.items():
        if term not in dictionary:
            continue
        term_postings = load_postings_list(postings_file, dictionary, term)
        idf = dictionary[term][0]
        for doc_id, tf_d in term_postings:
            scores[doc_id] += tf_d * idf
    normalized_scores = sorted(((doc_id, score / vector_lengths[doc_id])
                                for doc_id, score in scores.items()),
                               key=lambda x: x[1],
                               reverse=True)
    normalized_scores = [*filter(lambda x: x[1] > THRESHOLD, normalized_scores)]
    output = LinkedList()
    output.extend(normalized_scores)
    return output


def query_to_vector(query: str) -> Counter:
    return Counter([normalise(x) for x in query.split(' ')])


def rocchio_algorithm(
        query: Dict[str, int],
        relevant_doc_ids: List[str],
        docs_vector: Dict[str, Tuple[int, int]],
        alpha: int,
        beta: int) -> Dict[str, int]:
    relevant_docs_sum = sum([Counter(docs_vector[doc_id]) 
                            for doc_id in relevant_doc_ids], Counter())
    relevant_docs_centroid = {doc_id: beta * count / len(relevant_doc_ids)
                              for doc_id, count in relevant_docs_sum.items()}
    normalized_query = {k: alpha * v for k, v in query.items()}
    return dict(Counter(normalized_query) + Counter(relevant_docs_centroid))
