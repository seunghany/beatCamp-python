import sys
sys.path.insert(0, '/Users/seung/SbaProjects/beatCamp-python')
from titanic.entity import Entitity
from titanic.service import Service


class Controller:
"""
PassengerId  고객ID,
Survived 생존여부,  --> 머신러닝 모델이 맞춰야 할 답 
Pclass 승선권 1 = 1등석, 2 = 2등석, 3 = 3등석,
Name,
Sex,
Age,
SibSp 동반한 형제, 자매, 배우자,
Parch 동반한 부모, 자식,
Ticket 티켓번호,
Fare 요금,
Cabin 객실번호,
Embarked 승선한 항구명 C = 쉐브루, Q = 퀸즈타운, S = 사우스햄튼
"""

    def __init__(self):
        self.entity = Entity()
        self.service = Service()


    def modeling(self,train, test):
        service = self.service
        this = self.preprocessing(train, test)
        print(f'훈련 컬럼: {this.train.colums}')
        this.label = service.create_label(this)
        this.train = service.create_train(this)


    def preprocessing(self, train, test):
        service = self.service
        this = self.entity
        this.train = service.new_model(train) # payload
        this.test = service.new_model(test) # payload 
        this.id = this.test['PassengerId] # machine 에게는 이것이 question 이 됩니다
        print(f'드롭 전 변수 : {this.train.columns}')
        this = service.drop_feature(this, 'Cabin')
        this = service.drip_feature(this, 'Ticket')
        print(f'드롭 후 변수 : {this.train.columns}')
        return this


        
    def learing(self):  # evaluation과 합친다.
        pass
    def submit(self): # 파일로 저장
        pass

if __name__ =='__main__':
    ctrl = Controller()
    ctrl.modeling('train.csv', 'test.csv')