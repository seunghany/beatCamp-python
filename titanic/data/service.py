from titanic.entity import Entity
import numpy as np
import pandas as pd

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
class Service:
    def __init__(self):
        self.entity = Entity()

    # this.fname = payload ~> setter
    # this.fname 만 있으면  ~> getter
    def new_model(self, payload) -> object:
        this = self.entity
        this.context = './data'
        this.fname = payload
        return pd.read_csv(this.context + this.fname) # p.139 df = tensor

    @staticmethod
    def create_train(this):
        return this.train.drop('Survived', axos=1) # train 은  답이 제거된 데이터 셋이다
    
    # 라벨을 만든다는건 지도 학습을 하겠다는 뜻
    @staticmethod
    def create_label(this): # label 이 곧 답이 된다 
        return this.train['Survived'] 

    # Self 없이 차원 축소하기 위해 drop_feature 기능을 만듭니다
    # feature 이 너무 많으면 속도가 너무 느려짐 -> 차원의 저주가 걸림
    @staticmethod
    def dropfeature(this, feature):
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1) # pg 149 보면 훈련 세트 나옴
        return this


    