from typing import Tuple, List, Dict
from enum import Enum


class BooleanConstants(Enum):
    BOOLEAN = 'boolean'
    FREE_TEXT = 'free_text'
    PHRASE = 'phrase'
    NON_PHRASE = 'nonphrase'


def filter_docs_by_boolean_query(parsed_query: Tuple[str, List[Tuple[str, str]]],
                                 documents: List[int],
                                 dictionary: Dict[str, List],
                                 bitriword_dictionary: Dict[Tuple, List]) -> List[int]:
    """
    Filters the query by boolean AND logic if the query provided is a boolean query. Returns all documents if a
    free text query is provided.

    :param parsed_query: A Tuple containing ('boolean' | 'free text', <query>).
        <query> is a List of Tuples with the form (<'phrase' | 'nonphrase', <term>)
        where <term> is a query term/phrase.
    :return: A list of document IDs.
    """
    is_boolean_query = parsed_query[0] == BooleanConstants.BOOLEAN
    query = parsed_query[1]

    if not is_boolean_query:
        return documents

    for term_type, phrase_or_term in query:
        if term_type == BooleanConstants.PHRASE:
            phrase = phrase_or_term
            documents = documents
        elif term_type == BooleanConstants.NON_PHRASE:
            term = phrase_or_term
            documents = documents

    return documents
