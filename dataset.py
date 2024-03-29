import pandas as pd

class Dataset:
    def __init__(self, filename):
        self.dataset = pd.read_csv(filename)
        self.df = pd.DataFrame(self.dataset)
