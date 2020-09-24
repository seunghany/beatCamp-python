from titanic.entity import Entity
import numpy as np
import pandas as pd
# https://github.com/seunghany/beatCamp-python.git
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
        this.fname = payload
        return pd.read_csv(this.context+'/titanic/' + this.fname) # p.139 df = tensor


    @staticmethod
    def create_train(this):
        return this.train.drop('Survived', axos=1)  # train 은  답이 제거된 데이터 셋이다

    # 라벨을 만든다는건 지도 학습을 하겠다는 뜻
    @staticmethod
    def create_label(this):
        return this.train['Survived']  # label 이 곧 답이 된다.

    # Self 없이 차원 축소하기 위해 drop_feature 기능을 만듭니다
    # feature 이 너무 많으면 속도가 너무 느려짐 -> 차원의 저주가 걸림
    @staticmethod
    def drop_feature(this, feature):
        this.train = this.train.drop([feature], axis=1)
        this.test = this.test.drop([feature], axis=1)  # pg 149 보면 훈련 세트 나옴
        return this

    # order
    # ordinal(순서), numeric(숫자), norminal(이름)
    @staticmethod
    def pclass_ordinal(this) -> object:
        return this

    @staticmethod
    def title_nominal(this) -> object:
        combine = [this.train. this.test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Mr')
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)  # for unknown
        this.train = this.train
        this.test = this.test

        return this

    @staticmethod
    def sex_nominal(this) -> object:
        # male = 0, female = 1
        combine = [this.train, this.test]  # combine two set of data
        sex_mapping = {'male': 0, 'female': 1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        this.train = this.train  # overiding
        this.test = this.test

        # this.train['Sex'] = this.train['Sex'].map({'male':0, 'female':1})
        # this.test['Sex'] = this.test['Sex'].map({'male':0, 'female':1})
        return this

    @staticmethod
    def age_ordinalt(this) -> object:
        train = this.train  # this 를 줄이기 위한 변수 처리
        test = this.test

        # Stage 1 : Fill NaN (missing file)
        train['Age'] = train['Age'].fillna(-0.5)  # 아직 나이를 모르니 일단 놔두자
        # age 는 평균을 넣기도 애매하고, 다수결을 넣기도 너무 근거가 없다..
        # 특히 age 는 생존율 판단에서 가중치(weight) 상당하므로 디테일한 접근이 필요합니다
        # 나이를 모르는 승객은 모르는 상태로 처리해야 값의 왜곡을 줄일수 있으므로
        # -0.5 라는 값을 넣어서 anomaly 처리해 없애 버릴려고 합니다
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]  # 총 9 # 범위
        # [] 에 있으니 이것은 변수명 이라고 생각하면 됩니다.
        labels = ['Unkown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']  # 총 8
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unkown',
            1: 'Baby',
            2: 'Child',
            3: 'Teenager',
            4: 'Student',
            5: 'Young Adult',
            6: 'Adult',
            7: 'Senior'
        }  # 이렇게 [] -> {}

        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]

        age_mapping = {
            'Unkown': 0,
            'Baby': 1,
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'Young Adult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

        this.train = train
        this.test = test
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
        this.test = this.test.fillna({'FareBand': 1})
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
        this.train['Embarked'] = this.train['Embarked'].map({'S': 1, "C": 2, "Q": 3})
        this.test['Embarked'] = this.train['Embarked'].map({'S': 1, "C": 2, "Q": 3})

        return this

    # Learning Algorithm 중에서 dtree, rforest, nb, knn svm 이것을 대표로 사용하겠습니다.

    def accuracy_by_dtree(self, this):
        pass

    def accuracy_by_rforest(self, this):
        pass

    def accuracy_by_nb(self, this):
        pass

    def accuracy_by_knn(self, this):
        pass

    def accuracy_by_svm(self, this):
        pass
