# relevant_docs: {docid: Counter{str: Int}}
# query: {term (str) : count (Int)}

# Constants
ALPHA = 1.0
BETA = 0.75


def rocchio(query, relevant_docs, alpha, beta):
    normalized_query = {k: alpha * v for k, v in query.items()}
    relevant_docs_centroid = {*filter(lambda x: x in query.keys(), {k: beta * v / len(relevant_docs) for k, v in sum(relevant_docs.values())})}
    return normalized_query + relevant_docs_centroid
