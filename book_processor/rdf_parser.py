__all__ = ['RDFParser', 'RDFBookData', 'Author']

import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))

import logging
import pickle
import re
import tarfile
import xml.etree.ElementTree as elemTree
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from notebook_utils.constants import get_dir

logging.basicConfig(filename="rdf_parser.log", filemode="w", format="%(asctime)s::%(levelname)s - %(message)s", level=logging.INFO)

sys.path.append("pydevd-pycharm.egg")

NS = {"rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "cc": "http://web.resource.org/cc/",
      "dcam": "http://purl.org/dc/dcam/",
      "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
      "dcterms": "http://purl.org/dc/terms/",
      "pgterms": "http://www.gutenberg.org/2009/pgterms/"}


class RDFParser:
    def __init__(self, rdf_path: str, save_path: str):
        self.rdf_path = rdf_path
        self.save_path = save_path
        self.books: List[RDFBookData] = []

    def extract_data(self):
        print(f"Extracting book data from {self.rdf_path}")
        tar = tarfile.open(self.rdf_path)
        members = tar.getmembers()
        for member in tqdm(members):
            b_num = member.name.split("/")[2]
            logging.info(f"Parsing {b_num}")
            f = tar.extractfile(member)
            book_data = RDFBookData(f, b_num, NS, self.save_path)
            self.books.append(book_data)
            # book_data.save()

    def write_csv(self):
        book_data = []
        for book in self.books:
            names = ';'.join([author.name for author in book.authors])
            birth_year = min([author.birth_year for author in book.authors]) if len(book.authors) > 0 else -1
            death_year = max([author.death_year for author in book.authors]) if len(book.authors) > 0 else -1
            book_data.append({"Book #": book.book_number, "Language": book.language,
                              "Title": book.title, "Authors": names,
                              "Min Pub Year": birth_year, "Max Pub Year": death_year,
                              "Subjects": ';'.join(book.subjects), "Genres": ';'.join(book.genres),
                              "Downloads": book.downloads})
        df = combine_duplicates(pd.DataFrame(book_data))
        csv = open(self.save_path, 'w+', encoding="utf-8", newline='')
        df.to_csv(csv, index=False)


class RDFBookData:
    def __init__(self, f: tarfile.ExFileObject, book_number: str, ns: Dict, save_path: str):
        self.ns = ns
        self.save_path = str(Path(save_path).joinpath(book_number))
        self.tree: elemTree = elemTree.fromstring(f.read())

        self.authors: List[Author] = []
        self.book_number: str = book_number
        self.downloads: int = 0
        self.genres: List = []
        self.language: str = ""
        self.subjects: List = []
        self.title: str = "no title"

        self.get_data()

    def __repr__(self):
        return f"{self.book_number}: {self.title} , {self.downloads}"

    def __str__(self):
        return f"{self.book_number}: {self.title} , {self.downloads}"

    def get_data(self):
        for item in self.tree.findall(".//dcterms:creator", self.ns):
            self.authors.append(Author(item, self.book_number, self.ns))

        try:
            self.downloads = int(self.tree.find(".//pgterms:downloads", self.ns).text)
        except (AttributeError, TypeError) as e:
            logging.error(f"B# {self.book_number} -- No download count", exc_info=True)

        for item in self.tree.findall(".//pgterms:bookshelf", self.ns):
            try:
                self.genres.append(item.find(".//rdf:value", self.ns).text)
            except (AttributeError, TypeError) as e:
                logging.error(f"B# {self.book_number} -- Issue parsing genre, genres = {self.genres}", exc_info=True)

        for item in self.tree.findall(".//dcterms:subject", self.ns):
            try:
                self.subjects.append(item.find(".//rdf:value", self.ns).text)
            except (AttributeError, TypeError) as e:
                logging.error(f"B# {self.book_number} -- Issue parsing subject, subjects = {self.subjects}", exc_info=True)

        try:
            self.language = self.tree.find(".//dcterms:language/rdf:Description/rdf:value", self.ns).text
        except (AttributeError, TypeError) as e:
            logging.error(f"B# {self.book_number} -- No language", exc_info=True)

        try:
            self.title = self.tree.find(".//dcterms:title", self.ns).text
        except (AttributeError, TypeError) as e:
            logging.error(f"B# {self.book_number} -- No title", exc_info=True)

    def save(self):
        with open(self.save_path, "wb+") as f:
            try:
                pickle.dump(self, f)
            except MemoryError:
                print(f"There was a MemoryError when dumping books to {self.save_path}")


class Author:
    def __init__(self, xml_item, book_number: str, ns: Dict):
        self.name = "name"
        self.birth_year: int = -1
        self.death_year: int = -1

        self.set_lifespan(xml_item, book_number, ns)
        self.set_name(xml_item, book_number, ns)

    def set_lifespan(self, xml_item, book_number: str, ns: Dict):
        death_date = xml_item.find(".//pgterms:deathdate", ns)
        try:
            if len(death_date) > 4:
                self.death_year = get_year(death_date.text)
            else:
                self.death_year = int(death_date.text)
        except (AttributeError, TypeError) as e:
            logging.error(f"B# {book_number} -- No author death date", exc_info=True)

        birth_date = xml_item.find(".//pgterms:birthdate", ns)
        try:
            if len(birth_date) > 4:
                self.birth_year = get_year(birth_date.text)
            else:
                self.birth_year = int(birth_date.text)
        except (AttributeError, TypeError) as e:
            logging.error(f"B# {book_number} -- No author birth date", exc_info=True)

    def set_name(self, xml_item, book_number: str, ns: Dict):
        name = xml_item.find(".//pgterms:name", ns)
        try:
            self.name = name.text
        except (AttributeError, TypeError) as e:
            logging.error(f"B# {book_number} -- No author name", exc_info=True)

    def __repr__(self):
        return f"{self.name} ({self.birth_year} - {self.death_year})"

    def __str__(self):
        return f"{self.name} ({self.birth_year} - {self.death_year})"


def get_year(date: str):
    year = int(re.findall("\\d\\d\\d\\d", date, re.IGNORECASE)[0])
    return year


def combine_duplicates(rdf_df: pd.DataFrame):
    print("Combining duplicate books...")
    groups = rdf_df.groupby(by=["Title", "Authors"], as_index=False)
    to_remove = []
    for name, group in tqdm(groups):
        if len(group) > 1:
            keep = group[group["Genres"].notnull()]
            if len(keep) > 1:
                keep = keep[keep["Downloads"] == keep["Downloads"].max()]
            if len(keep) == 0:
                keep = group[group["Downloads"] == group["Downloads"].max()]
            rdf_df.at[keep.index.values[0], "Downloads"] = group["Downloads"].sum()
            for idx in list(group.loc[~group.index.isin(keep.index.values)].index):
                to_remove.append(idx)
    return rdf_df.loc[~rdf_df.index.isin(to_remove)].reset_index(drop=True)


if __name__ == "__main__":
    root_path = get_dir(Path().cwd(), "README.md")
    default_rdf_path = str(root_path.joinpath("data", "rdf_data", "rdf-files.tar"))
    default_save_path = str(root_path.joinpath("data", "rdf_data", "extracted_rdf_data.csv"))

    p = ArgumentParser()
    p.add_argument("-rdf_path", type=str, default=default_rdf_path, help="Path where RDF data is located")
    p.add_argument("-save_path", type=str, default=default_save_path, help="Path where extracted data will be saved as a csv")

    args = p.parse_args()
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)

    rdf_parser = RDFParser(args.rdf_path, args.save_path)
    rdf_parser.extract_data()
    rdf_parser.write_csv()
