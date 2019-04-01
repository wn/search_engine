from typing import Tuple, List, Dict, IO, Iterable
from enum import Enum
import pickle
from functools import reduce
from .data_structures import LinkedList


class BooleanConstants(Enum):
    BOOLEAN = 'boolean'
    FREE_TEXT = 'free_text'
    PHRASE = 'phrase'
    NON_PHRASE = 'nonphrase'

def perform_and(operand_a: LinkedList, operand_b: LinkedList) -> LinkedList:
    """Returns all ids that are ids of operand a and operand b. Copied from HW2."""
    result = LinkedList()
    operand_a, operand_b = operand_a.get_head(), operand_b.get_head()
    while operand_a is not None and operand_b is not None:
        if operand_a.value == operand_b.value:
            result.append(operand_a.value)
            operand_a = operand_a.next()
            operand_b = operand_b.next()
        elif operand_a.value < operand_b.value:
            if operand_a.skip() and operand_a.skip().value <= operand_b.value:
                operand_a = operand_a.skip()
            else:
                operand_a = operand_a.next()
        elif operand_b.skip() and operand_b.skip().value <= operand_a.value:
            operand_b = operand_b.skip()
        else:
            operand_b = operand_b.next()
    return result


def perform_boolean_query(query_pairs: List[Tuple[str, str]],
                          tfidf_dictionary: Dict[str, Tuple],
                          bitriword_dictionary: Dict[str, Tuple],
                          postings_file: IO) -> LinkedList:
    """
    Returns a LinkedList of documents that satisfy a purely conjunctive boolean query.

    :param query_pairs: A Tuple containing a List of Tuples with the form (<'phrase' | 'nonphrase', <term>)
        where <term> is a query term/phrase.
    :param tfidf_dictionary the TF-IDF dictionary
    :param bitriword_dictionary the biword/triword dictionary
    :param postings_file the file descriptor for the postings list file.
    :return: A LinkedList of document IDs.
    """
    def get_postings_list_length(query_pair: Tuple[str, str]) -> int:
        """Helper function to get a postings list length of a phrase/term. Assumes that phrase exists."""
        term_type, phrase = query_pair
        if term_type == BooleanConstants.PHRASE:
            return bitriword_dictionary[phrase][0]
        elif term_type == BooleanConstants.NON_PHRASE:
            return tfidf_dictionary[phrase][0][1]

    def get_postings_list(query_pair: Tuple[str, str]) -> LinkedList:
        """Helper function to get a postings list length of a phrase/term. Returns empty Linked"""
        term_type, phrase = query_pair
        if term_type == BooleanConstants.PHRASE:
            _, offset, length = bitriword_dictionary[phrase]
        else:
            _, offset, length = tfidf_dictionary[phrase]
        postings_file.seek(offset)
        return pickle.loads(postings_file.read(length))

    def phrase_exists(query_pair: Tuple[str, str]) -> bool:
        term_type, phrase = query_pair
        if term_type == BooleanConstants.PHRASE and phrase in bitriword_dictionary:
            return True
        elif term_type == BooleanConstants.NON_PHRASE and phrase in tfidf_dictionary:
            return True
        else:
            return False

    if any(not phrase_exists for query_pair in query_pairs):
        return LinkedList()

    # Optimization for faster AND computation
    list.sort(query_pairs, key=get_postings_list_length)

    # Generate a list of Postings lists
    postings_lists: Iterable = map(get_postings_list, query_pairs)

    return reduce(perform_and, postings_lists)


