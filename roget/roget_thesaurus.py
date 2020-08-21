import re
import sqlite3
from pathlib import Path

import pandas as pd


class RogetThesaurus:
    def __init__(self, path_to_csv: Path = Path.cwd().joinpath("roget_thesaurus.csv"), create_db: bool = False):
        self.path: Path = path_to_csv
        self.roget_df = pd.read_csv(self.path)
        self.roget_df["Category"] = self.roget_df["Category"].apply(lambda x: re.sub("\\d*\\s*", "", x))
        # self.roget_df["Count"] = 0
        self.roget_df["Words"] = self.roget_df["Words"].apply(lambda x: x.split("|"))
        if create_db:
            self.roget_df.to_sql("roget_thesaurus", sqlite3.connect("roget_thesaurus.db"), if_exists="replace", index=False)
        # self.tree = RogetTree(self.roget_df)
        self.roget_dict = self.build_roget_dict()

    def build_roget_dict(self):
        roget_dict = {class_: {} for class_ in list(self.roget_df["Class"].unique())}
        for class_ in list(roget_dict.keys()):
            sections = list(self.roget_df["Section"].loc[self.roget_df["Class"] == class_].unique())
            for section in sections:
                categories = list(self.roget_df["Category"].loc[self.roget_df["Section"] == section])
                for category in categories:
                    words = list(set([word for word_list in list(self.roget_df["Words"].loc[
                                                                     (self.roget_df["Class"] == class_) &
                                                                     (self.roget_df["Section"] == section) &
                                                                     (self.roget_df["Category"] == category)].values)
                                      for word in word_list]))
                    if section in roget_dict[class_].keys():
                        roget_dict[class_][section].update({category: words})
                    else:
                        roget_dict[class_].update({section: {category: words}})
        return roget_dict

    def get_categories(self, word: str):
        keys = []
        for class_ in list(self.roget_dict.keys()):
            for section, categories in self.roget_dict[class_].items():
                for category, words in self.roget_dict[class_][section].items():
                    if word in words:
                        keys.append(category)
        return keys
