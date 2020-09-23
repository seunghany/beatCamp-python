 # 0920_naver_movie_ranking.py

from bs4 import BeautifulSoup
from urllib.request import urlopen

import os, shutil

myfolder = 'C:\\Users\\bumsu\\SbaProject\\test01\\'

def saveFile(movie_src, movie_name):
    image_open = urlopen(movie_src)
    filename = myfolder + movie_name + '.jpg'
    myfile = open(filename, mode='wb')
    myfile.write(image_open.read())
    myfile.close()

try:
    if not os.path.exists(myfolder):
        os.mkdir(myfolder)

except FileExistsError as err :
    print(err)


myurl = 'https://movie.naver.com/movie/running/current.nhn'

response = urlopen(myurl)

soup = BeautifulSoup(response, 'html.parser')


mytarget_title = soup.findAll('div', attrs={'class':'thumb'})

mytarget_star = soup.findAll('dd', attrs={'class':'star'})

mylist0 = []
mylist1 = []

for aaa0 in mytarget_title:
    movie_name = aaa0.find('img').attrs['alt']
    movie_name = movie_name.replace('?', '').replace(':', '')
    movie_src_full = aaa0.find('img').attrs['src']
    movie_src = movie_src_full.replace('?type=m99_141_2', '')
    sublist = []

    sublist.append(movie_name)
    sublist.append(movie_src)

    mylist0.append(sublist)

    saveFile(movie_src, movie_name)

for aaa1 in mytarget_star:
    myhref0 = aaa1.find('a')
    movie_point_full = myhref0.find('span', attrs={'class':'num'})
    movie_point = movie_point_full.contents

    movie_reserve_full = aaa1.find('div', attrs={'class':'star_t1 b_star'})

    try:
        movie_reserve = movie_reserve_full.find('span', attrs={'class':'num'}).contents

    except AttributeError as err:
        # print(err)
        movie_reserve = '미개봉'

    sublist = []

    sublist.append(movie_point)
    sublist.append(movie_reserve)

    mylist1.append(sublist)

# print(mylist0)
# print('-'*30)
# print(mylist1)

from pandas import DataFrame
import pandas as pd

mycolumns0 = ['제목', '스크린샷']
mycolumns1 = ['별점', '예매율']
myindex = range(0, len(mylist0))
myframe0 = DataFrame(mylist0, index=myindex, columns=mycolumns0)
myframe1 = DataFrame(mylist1, index=myindex, columns=mycolumns1)

myframe = pd.concat([myframe0, myframe1],axis=1)
filename = '0920_naver_movie_ranking.csv'
myframe.to_csv(filename, encoding='utf-8')
print(filename + ' 파일로 저장됨')

print('finished')
© 2020 GitHub, Inc.