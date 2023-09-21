# ###
# This TableExtract Class defines the load data functions which extract data from raw tables
# ###

from abc import ABC, abstractmethod


class TableExtract(ABC):

    @staticmethod
    @abstractmethod
    def load_data(table):
        pass

    @staticmethod
    @abstractmethod
    def load_all_data():
        pass
