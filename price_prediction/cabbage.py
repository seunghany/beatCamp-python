import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.file_handler import FileReader
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '/Users/seung/SbaProjects/beatCamp-python')
from dataclasses import dataclass

# util file path-->
# C:\Users\seung\SbaProjects\beatCamp-python\util\file_handler.py


class Model:

    #  year, avgTemp, minTemp, maxTemp, rainFall, avgPrice
    #  20100101, -4.9, -11, 0.9, 0, 2123
    #  멤버 변수
    year: int = 0
    avgTemp: float = 0.0
    minTemp: float = 0.0
    maxTemp: float = 0.0
    rainFall: int = 0.0
    avgPrice: int = 0



    def __init__(self):
        self.fileReader = FileReader()
        self.context = '/Users/seung/SbaProjects/beatCamp-python/price_prediction/data/'

    def new_model(self, payload) -> object:
        this = self.fileReader
        this.context = self.context
        this.fname = payload
        return pd.read_csv(this.context + this.fname, sep=',')

    def create_tf(self, payload):
        xy = np.array(payload, dtype=np.float32)
        x_data = xy[:,1:-1] # feature
        y_data = xy[:,[-1]] # price
        X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
        Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random.normal([4, 1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')
        hyposthesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hyposthesis - Y))
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hyposthesis, train],
                                        feed_dict={X: x_data, Y: y_data})
            if step % 500 == 0:
                print(f'# {step} 손실비용: {cost_} ')
                print(f'- 배추가격 : {hypo_[0]}')

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, self.context+'saved_model.ckpt')
        print('저장완료')
    
    def test(self):
        self.avgPrice = 100
        return self.avgPrice

    def service(self):
        X = tf.compat.v1.placeholder(tf.float32, shape[None, 4])
        # year, avgTemp, minTemp, maxTemp, rainFall, avgPrice
        # 에서 avgTemp, minTemp, maxTemp, rainFall 입력 받겠다
        # year 는 모델에서 필요없는 값 --> 상관관계 없음
        # avg 는 얻고자 한ㄴ 답. 종속 변수
        # avgTemp, minTemp, maxTemp, rainFall 는 종속변수를 결정하는 독립변수
        # 그리고 avgPrice 를 결정하는 요고로 사용되는 파라미터 (중요)
        # 이제 우리는 통계와 확률로 들어가야 합니다. 용어를 먼저 정리 합시다.
        W = tf.Variable(tf.random.normal([4, 1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variable_initializer())
            saver.restore(sess, self.context+'saved_model.ckpt')
            data = [[self.avgPrice, self.minTemp, self.maxTemp, self.rainFall],]
            arr = np.array(data, dtype = np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict[0])
            return int(dict[0])
if __name__ == '__main__':
    m = Model()
    # dframe = m.new_model('price_data.csv')
    # print(dframe.head())
    # m.create_tf(dframe)
    print(m.test())