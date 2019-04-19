from typing import Tuple, List, Dict, BinaryIO, Union, cast

from data_structures import LinkedList, TokenType
from phrasal_retrieval import retrieve_phrase
from search_helpers import load_postings_list


def perform_and(operand_a: LinkedList[int],
                operand_b: LinkedList[int]) -> LinkedList:
    """
    Returns all ids that are ids of operand a and operand b. Copied from HW2.
    """
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


def perform_boolean_query(
        tokens: List[Tuple[str, Union[List[str], str]]],
        dictionary: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]],
        postings_file: BinaryIO) -> LinkedList:
    """
    Returns a LinkedList of documents that satisfy a purely conjunctive boolean
    query.

    :param tokens: A List containing Tuples with the form
        (<'phrase' | 'nonphrase', <term>) where <term> is a query term/phrase
        -- phrases are Lists, single terms are strings.
    :param dictionary the combined TF-IDF and positional index dictionary
    :param postings_file the read-only binary file descriptor for the postings
        list file.
    :return: A LinkedList of document IDs that satisfy the boolean query.
    """

    def get_idf(token: Tuple[str, Union[List[str], str]]) -> float:
        """
        Helper function to get the IDF of a phrase/term.

        The IDF of a phrase is estimated by summing up the IDF of the
        individual terms. This works out as phrases usually produce small
        resultant postings lists, and a phrase with rare terms (high IDF) will
        produce smaller postings lists than a phrase with common terms.
        """
        term_type, phrase = token
        if term_type == TokenType.PHRASE:
            phrase = cast(List[str], phrase)
            # Returns the sum of the idfs of each term
            # (first item in the dictionary tuple)
            return sum(
                map(
                    lambda term: dictionary[term][0]
                    if term in dictionary else 0, phrase))
        elif term_type == TokenType.NON_PHRASE:
            term = cast(str, phrase)
            return dictionary[term][0] if term in dictionary else 0

    def get_postings_list(term_type: str,
                          phrase: Union[List[str], str]) -> LinkedList:
        """
        Helper function to get a postings list length of a phrase/term.
        Returns empty LinkedList if phrase does not exist.
        """
        if term_type == TokenType.PHRASE:
            return retrieve_phrase(dictionary, postings_file, phrase)
        elif term_type == TokenType.NON_PHRASE:
            term = cast(str, phrase)
            return load_postings_list(postings_file, dictionary, term)

    # Guard against empty tokens list
    if not tokens:
        return LinkedList()

    # Optimization for faster AND computation -- do the rarer term.
    list.sort(tokens, key=get_idf, reverse=True)

    # Generate a list of Postings lists
    resultant_list: LinkedList[int] = get_postings_list(*tokens[0])

    # Successively use AND on the tokens' postings lists
    for token in tokens[1:]:
        # Short circuit for empty LinkedList --
        # cannot be done easily when using `reduce`
        if not resultant_list:
            break
        resultant_list = perform_and(resultant_list, get_postings_list(*token))

    return resultant_list
