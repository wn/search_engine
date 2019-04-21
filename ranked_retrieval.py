from typing import Tuple, List, Dict, BinaryIO
from collections import Counter, defaultdict
from nltk.corpus import wordnet

from data_structures import LinkedList
from search_helpers import normalise, load_postings_list, load_document_vector

# Variables for tuning
ALPHA = 1.0
BETA = 0.75
THRESHOLD = 0
QUERY_EXPANSION = True
RELEVANT_FEEDBACK = True


def get_relevant_docs(
        query: str,
        dictionary: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]],
        vector_lengths: Dict[str, float], relevant_doc_ids: List[str],
        document_vectors_dictionary: Dict[str, Tuple[int, int]],
        postings_file: BinaryIO) -> LinkedList:
    query_vector = query_to_vector(query)
    if QUERY_EXPANSION:
        query_vector = query_expansion(query_vector)
    if RELEVANT_FEEDBACK:
        query_vector = rocchio_algorithm(query_vector, relevant_doc_ids,
                                         document_vectors_dictionary, ALPHA, BETA,
                                         postings_file)
    scores = defaultdict(float)
    for term, factor in query_vector.items():
        if term not in dictionary:
            continue
        term_postings = Counter(load_postings_list(postings_file, dictionary, term))
        idf = dictionary[term][0]
        for doc_id, tf_d in term_postings:
            scores[doc_id] += factor * tf_d * idf
    normalized_scores = sorted(((doc_id, score / vector_lengths[doc_id])
                                for doc_id, score in scores.items()),
                               key=lambda x: x[1],
                               reverse=True)
    relevant_docs = (x[0] for x in normalized_scores if x[1] > THRESHOLD)
    output = LinkedList()
    output.extend(relevant_docs)
    return output


def query_to_vector(query: str) -> Counter:
    return Counter(normalise(x) for x in query.split(' '))


def rocchio_algorithm(query: Dict[str, int], relevant_doc_ids: List[str],
                      docs_vector: Dict[str, Tuple[int, int]], alpha: int,
                      beta: int, postings_file: BinaryIO) -> Dict[str, float]:
    relevant_docs_sum = Counter()
    for doc_id in relevant_doc_ids:
        relevant_docs_sum += load_document_vector(doc_id, postings_file,
                                                  docs_vector)
    relevant_docs_centroid = Counter({
        doc_id: beta * count / len(relevant_doc_ids)
        for doc_id, count in relevant_docs_sum.items()
    })

    normalized_query = Counter({k: alpha * v for k, v in query.items()})
    return dict(normalized_query + relevant_docs_centroid)


def query_expansion(query):
    result = defaultdict(float)
    for term, count in query:
        synonyms = []
        syn_term = wordnet.synsets(term)[0]
        for synset in wordnet.synsets(term)[1:]:
            factor = wordnet_score_factor(syn_term, synset)
            if factor:
                new_term = synset.lemmas()[0].name()
                result[new_term] += factor * count
    return dict(Counter(result) + Counter(query))


def wordnet_score_factor(first_word, word):
    return first_word.wup_similarity(word)
