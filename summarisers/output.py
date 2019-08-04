import json
import os

JSON_OUTDIR = "./data/json/"


def json_output(docid, document):
    """Output a json version of the processed document

    Args:
        document (objet): document to store
        docid (int): identifier of the document

    """
    for sentence in document:
        sentence.pop("embedding")
        sentence.pop("l_sentence")

    outfilename = os.path.join(JSON_OUTDIR, f"{docid}.json")
    with open(outfilename, "w") as f:
        json.dump(document, f, indent=2)
