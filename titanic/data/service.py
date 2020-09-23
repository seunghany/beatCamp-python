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
        return this.train.drop('Survived', axos=1)  # train 은  답이 제거된 데이터 셋이다
    
    # 라벨을 만든다는건 지도 학습을 하겠다는 뜻
    @staticmethod
    def create_label(this):
        return this.train['Survived']  # label 이 곧 답이 된다 

    # Self 없이 차원 축소하기 위해 drop_feature 기능을 만듭니다
    # feature 이 너무 많으면 속도가 너무 느려짐 -> 차원의 저주가 걸림
    @staticmethod
    def dropfeature(this, feature):
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1) # pg 149 보면 훈련 세트 나옴
        return this

    # order
    # ordinal(순서), numeric(숫자), norminal(이름)
    @staticmethod
    def pclass_ordinal(this) -> object:
        return this

    @staticmethod
    def name_nominal(this) -> object:
        return this

    @staticmethod
    def sex_nominal(this) -> object:
        # male = 0, female = 1
        this.train['Sex'] = this.train['Sex'].map({'male':0, 'female':1})
        this.test['Sex'] = this.test['Sex'].map({'male':0, 'female':1})
        return this

    @staticmethod
    def age_ordinalthis) -> object:
        return this
    
    @staticmethod
    def sibsp_numeric(this) -> object:
        return this

    @staticmethod
    def parch_numeric(this) -> object:
        return this
    
    @staticmethod
    def fareBand_nominal(this) -> object:  # 요금이 다양하니 Clustering 을 하기위한 준비
        this.train = this.train.fillna({'FareBand': 1})  # FareBand 는 없는 변수인데 추가함
        this.test = this.test.fillna({'FareBand' : 1})
        return this
    
    @staticmethod
    def fare_ordinal(this) -> object:
        this.train['FareBand'] = pd.qcut(this['Fare'], 4, labels={1, 2, 3, 4})
        this.test['FareBand'] = pd.qcut(this['Fare'], 4, labels={1, 2, 3, 4})
        
    @staticmethod
    def embarked_nominal(this) -> object:
        this.train = this.train.fillna({'Embarked': 'S'})  # S 가 가장 많이 쓰이므로 이것으로 빈칸을 채움
        this.test = this.test.fillna({"Embarked": "S"})  # 교과서 144pg 참고
        # 많은 머신러닝 라이브러리 클래스 레이블은 "정수" 로 인코딩 되어있을거라고 기대되고 있음
        # 교과서 146 pg 문자 blue = 0, green = 1, red = 2 로 치환 합니다.
        # 이런식으로 정수로 치환을 해서 사용 합니다.
        this.train['Embarked'] = this.train['Embarked'].map({'S' :1, "C" : 2, "Q": 3})
        return this
    
        return this
    
    