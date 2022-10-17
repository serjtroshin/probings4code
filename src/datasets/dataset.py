from abc import ABC, abstractmethod


class Dataset:
    @abstractmethod
    def __iter__(self):
        return None

    @property
    def type(self):
        return self._type

    def __len__(self):
        return 1e20
