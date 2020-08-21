import os
import unittest
from pathlib import Path

from tqdm import tqdm

from book_processor.gutenberg_processor import process_html
from notebook_utils.constants import get_dir


class MyTestCase(unittest.TestCase):
    def test_something(self):
        root_path = get_dir(Path().cwd(), "README.md")
        files_path = root_path.joinpath("data")
        
        if not os.path.exists(str(files_path.joinpath("processed html test files"))):
            os.makedirs(str(files_path.joinpath("processed html test files")))

        book_numbers = ["11", "1934", "521", "19"]
        titles = ["Alice’s Adventures in Wonderland", "Songs of Innocence and Songs of Experience",
                  "Robinson Crusoe", "The Song Of Hiawatha"]

        print("Processing new books...")
        for root, dirs, files in os.walk(str(files_path.joinpath("html test files"))):
            files.sort()
            with tqdm(total=len(files)) as pbar:
                for i in range(len(files)):
                    pbar.set_postfix_str(f" -- [{book_numbers[i]}, {titles[i]}")
                    f = open(str(files_path.joinpath("html test files", files[i])), "r+", encoding="utf-8")
                    processed = process_html(f, book_numbers[i])
                    with open(str(root_path.joinpath("processed html test files", f"{book_numbers[i]}.txt")), "w+") as f:
                        f.write(processed)
                    pbar.update(1)


if __name__ == '__main__':
    unittest.main()
