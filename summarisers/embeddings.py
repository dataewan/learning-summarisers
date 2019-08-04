from . import data
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from tqdm import tqdm
import random

EMBEDDING_FILENAME = "./data/embeddings.pkl"


def get_sentences(record, nlp):
    """Extract sentences from a piece of text.

    Args:
        record (dict): the document

    Returns: list of sentences, lower case sentences and indexes

    """
    text = record["text"]
    text = " ".join(text)
    doc = nlp(text)
    return [
        {
            "index": idx,
            "sentence": str(sentence),
            "l_sentence": str(sentence).lower(),
            "document_id": record["idx"],
        }
        for idx, sentence in enumerate(doc.sents)
    ]


def embedding_function(embed_module_url="https://tfhub.dev/google/elmo/2"):
    """An embedding function.

    Args:
        embed_module (str): URL that we get the embedder from

    Returns: function

    """
    with tf.Graph().as_default():
        embed = hub.Module(embed_module_url)
        sentences = tf.placeholder(tf.string)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()

    return lambda x: session.run(embeddings, {sentences: x})


def get_embeddings(sentences, embed):
    """Get the embedding for each sentence

    Args:
        sentences (list): list of sentences
        embed : embedding function

    Returns: list of sentences with embeddings added

    """
    for sentence in sentences:
        embedding = embed([sentence["l_sentence"]])
        sentence.update({"embedding": embedding})

    return sentences


def setup_nlp():
    """Create a spacy NLP object

    Returns: spacy nlp object

    """
    nlp = spacy.load("en_core_web_sm")
    return nlp


def save_embeddings(embeddings):
    """Pickle the embeddings

    Args:
        embeddings: sentences with embeddings
    """
    with open(EMBEDDING_FILENAME, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings():
    """Load the embeddings from pickle
    Returns: sentences with embeddings

    """
    with open(EMBEDDING_FILENAME, "rb") as f:
        return pickle.load(f)


def process_documents():
    """Extract sentence embeddings for each document in the corpus
    """
    records = data.get_records()
    records = random.sample(records, 40)
    nlp = setup_nlp()
    embed = embedding_function()
    sentences_w_embeddings = []
    for record in tqdm(records):
        sentences = get_sentences(record, nlp)
        sentences_w_embeddings.extend(get_embeddings(sentences, embed))

    save_embeddings(sentences_w_embeddings)

    return sentences_w_embeddings
