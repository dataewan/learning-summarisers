import sqlite3
import glob
import os
import pickle
import logging

DATA_DIR = "/home/ewan/data/bbc-datasets/bbc-news/bbc"
RECORDS_FILENAME = "./data/records.pkl"

logging.basicConfig(level=logging.DEBUG)


def find_files():
    globpath = os.path.join(DATA_DIR, "*/*.txt")
    return glob.glob(globpath)


def get_category(filename):
    # we want to get the penultimate part of the filename
    stub = os.path.split(filename)[0]
    return os.path.split(stub)[1]


def get_headline(lines):
    """Get the headline

    Args:
        lines (list of strings): lines in the file

    Returns: headline str

    """
    return lines[0].strip()


def get_text(lines):
    """Get the body text of the file

    Args:
        lines (list of strings): lines in the file

    Returns: list of strings

    """
    text = [i.strip() for i in lines[2:]]
    text = [
        i for i in text
        if i != ""
    ]
    return text


def process_body(filename):
    """Extract information from the body of the text file

    Args:
        filename (str): path

    Returns: (headline, text) both str

    """
    with open(filename, "r") as f:
        lines = f.readlines()
        headline = get_headline(lines)
        text = get_text(lines)

    return headline, text


def output(records):
    """Output records

    Kwargs:
        records: list of dicts
        records_filename (str): outfilename

    """
    with open(RECORDS_FILENAME, "wb") as f:
        pickle.dump(records, f)


def read_pickle():
    with open(RECORDS_FILENAME, "rb") as f:
        records = pickle.load(f)
    return records


def process():
    records = []
    files = find_files()
    for idx, filename in enumerate(files):
        category = get_category(filename)
        headline, text = process_body(filename)

        records.append(
            {
                "idx": idx,
                "category": category,
                "filename": filename,
                "headline": headline,
                "text": text,
            }
        )

    output(records)
    return records


def get_records():
    """Either read records from file or read fresh
    Returns: records 

    """
    if os.path.exists(RECORDS_FILENAME):
        records = read_pickle()
    else:
        records = process()

    return records
