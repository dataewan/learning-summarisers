from . import embeddings, output
import itertools
import numpy as np
import scipy.spatial.distance
import networkx as nx


def calculate_similarity(vec1, vec2):
    """Calculate the similarity between two vectors

    Args:
        vec1 (vector)
        vec2 (vector)

    Returns: float similarity score

    """
    return 1.0 - scipy.spatial.distance.cosine(vec1, vec2)


def form_similarity_matrix(sentence_embedding_vectors):
    """Create a similarity matrix from the embedding vectors.

    Args:
        sentence_embedding_vectors (list): list of the embeddings

    Returns: matrix

    """
    vectors = np.array(sentence_embedding_vectors)
    len_sentences = vectors.shape[0]

    similarity_matrix = np.zeros((len_sentences, len_sentences))

    permutations = list(itertools.permutations(range(0, len_sentences), 2))
    for pair in permutations:
        idx1, idx2 = pair
        vec1 = vectors[idx1][0]
        vec2 = vectors[idx2][0]

        similarity = calculate_similarity(vec1, vec2)

        similarity_matrix[idx1][idx2] = similarity

    return similarity_matrix


def get_pagerank_scores(similarity_matrix):
    """Calculate the pagerank of each sentence

    Args:
        similarity_matrix (matrix); similarity matrix of the sentences

    Returns: pagerank of each sentence

    """
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    return scores


def order_sentences(document, pagerank_scores):
    """Put an ordering on the sentences in the document

    Args:
        document (list): list of the document and sentences
        pagerank_scores (dict): dict of the sentences and their scores

    Returns: document, with an ordering of the sentences

    """
    scores = list(pagerank_scores.values())
    seq = sorted(scores, reverse=True)
    index = [seq.index(v) for v in scores]
    for idx, sentence in enumerate(document):
        sentence.update({"importance": index[idx]})
    return document


def summarise(document):
    """Find the most important sentences in the document

    Args:
        document (list): the document and sentences

    Returns: the document with importance scores for each sentence applied

    """
    embeddings = [s["embedding"] for s in document]
    similarity_matrix = form_similarity_matrix(embeddings)

    pagerank_scores = get_pagerank_scores(similarity_matrix)

    ordered_document = order_sentences(document, pagerank_scores)
    return ordered_document


def summarise_documents():
    """Summarise all the documents we've made embeddings for.
    """
    documents = embeddings.load_embeddings()

    keyfunc = lambda x: x["document_id"]

    for docid, document in itertools.groupby(documents, keyfunc):
        document = list(document)
        summarised = summarise(document)

        output.json_output(docid, summarised)
