 import sys
sys.path.insert(0, '/Users/seung/SbaProjects')
from titanic.entity import Entitity
from titanic.service import Service
class Controller:


    def __init__(self):
        self.entity = Entity()
        self.service = Service()


    def preprocessing(self, train, test):
        service = self.service
        this = self.entity
        this.train = service.new_model(train) #payload
        this.test = service.new_model(test)
        return this


    def modeling(self,train, test):
        service = self.service
        this = self.preprocessing(train, test)
        print(f'훈련 컬럼: {this.train.colums}')
        this.label =service.create_label(this)
        this.train = service.create_train(this)

        
    def learing(self):  # evaluation과 합친다.
        pass
    def submit(self): # 파일로 저장
        pass
if __name__ =='__main__':
    ctrl = Controller()
    ctrl.modeling('train.csv', 'test.csv')