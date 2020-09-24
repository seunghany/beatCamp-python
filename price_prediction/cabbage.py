import sys
sys.path.insert(0, '/Users/seung/SbaProjects/beatCamp-python')
from util.file_handler import FileReader
import padnas as pd
import numpy as np

class Model:
    def __init__(self):
        self.fileReader = FileReader()

    def new_model(self, payload) -> object:
        this = this.fileReader
        this.context = '/Users/seung/SbaProjects/beatCamp-python/price_prediction/data'
        this.fname = payload
        return pd.read_csv(this.context + this.fname)

    if __name__ == "__main__":
        m = Model()
        dframe = m.new_model('price_data.csv')
        print(dframe.head())
        

    # @staticmethod
    # def create_train(this):
    #     return this.train.drop('avgPrice', axis=1)  # train 은  답이 제거된 데이터 셋이다