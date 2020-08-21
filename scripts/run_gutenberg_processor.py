import os
import sys

sys.path.append(os.getcwd())

from multiprocessing import Pool
from notebook_utils.constants import NEW_GENRES
from book_processor.gutenberg_processor import process_gutenberg_files, process_rdf_df

if __name__ == "__main__":
    rdf_data_df, by_genre_dict = process_rdf_df()
    args = [(genre, rdf_data_df, "txt") for genre in NEW_GENRES]
    with Pool() as p:
        p.starmap(process_gutenberg_files, args)
