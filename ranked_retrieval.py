
def process_queries(dictionary, postings_file_location,
                    file_of_queries_location, file_of_output_location):
    """
    Process all the queries in the queries file.
    """
    with open(file_of_queries_location, 'r') as queries, \
            open(postings_file_location, 'rb') as postings_file, \
            open(file_of_output_location, 'w') as output_file:
        for query in queries:
            try:
                process_query(query, dictionary, postings_file, output_file)
            except Exception as e:
                output_file.write("\n")


def process_query(query, dictionary,lengths_dictionary, postings_file, output_file):
    """
    Calculates the cosine scores of the documents, get 10 documents with the
    highest scores and writes to file
    """
    scores = Counter()
    weighted_tfs = get_weighted_tfs(parse(query))
    for term, tf_q in weighted_tfs.items():
        if term not in dictionary:
            continue
        postings = load_postings(postings_file, dictionary, term)
        idf = dictionary[term][0]
        for doc, tf_d in postings:
            scores[doc] += tf_d * tf_q * idf
    for doc in scores.keys():
        scores[doc] = float(scores[doc]) / lengths_dictionary[doc]

    postings = " ".join(str(x) for x in retrieve_top_ten_scores(scores))
    output_file.write(postings + "\n")


def retrieve_top_ten_scores(scores):
    """
    Retrieve the top 10 postings by score. This is done using the heapq.
    The results are then sorted by decreasing score and then by increasing
    lexicographical order of terms.
    """
    score_list = heapq.nlargest(10, scores.items(), lambda x: (x[1], -x[0]))
    return (x[0] for x in score_list)

