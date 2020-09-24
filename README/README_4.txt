Variable (변하는 상태) vs Constant (변하지 않음)

그중에서는 variable 은 분류가 가능합니다
분류기준을 두고 나누는데

크제 2개로 분류를 하면
cate, norminal(=: name)
다시 cate 는 ordinal(=: order), numerical (=: number)

그래서 결국은,
ordinal, numeric, norminal
순서, 숫자, 이름 으로 나눌 수 있다

이곳 (확률 통계 코딩)은 정답 보다는 적합 하다라는 개념입니다

embarked 부터 가면,
교과서 138를 보면 누락딘 값 처리 방식이 나옵니다.
지금 이 embarked 를 지우면 안되고 (dropna 를 쓰면 안됩니다.)
139 페이지에 나오는 대체 하는 방식을 사용해야 합니다.

여기서 numm 값을 무엇으로 넣을 것인가?
평균 값을 넣자고 책에는 되있지만, 
embarked는 str 이기 때문에 평균을 구할 수 없다.
그래서 가장 많이 승선한 항구로 대체
물론 이게 통계를 왜곡 할 수 있다, 하지만
null 의 숫자가 적으니 크게 영향을 미치진 않는다.
하지만 null 값이 있으면, 그 데이터 자체를 사용 할 수 없으니 
차라리 조금 영향을 끼치더라고 사용하도록 합시다

@staticmethod
def sex_norminal(this)--> object
    # male = 0, female = 1
    this.train['Sex'] = this.train['Sex'].map({'male:0', 'female:1'})
    this.train['Sex'] = this.test['Sex'].map({'male:0', 'female:1'})

    코딩은 반복된 코드를 싫어합니다.
    for(), while() 이 존재하는 이유

    그래서 위 코드에서 반복을 피하기 위해서 (지도학습의 숙명인 train과 테스트 둘다 편집해야 하는 상황)
    다음과 같은 코드가 나옵니다.

    combine = [this.train, this.test]  # combine two set of data
        sex_mapping = {'male': 0, 'female': 1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        this.train = this.train  # overiding
        this.test = this.test