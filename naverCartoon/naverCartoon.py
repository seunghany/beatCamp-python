# naverCartoon.py
from bs4 import BeautifulSoup

# urlopen : 네트워크에 존재하는 주소를 열어 주는 함수
from urllib.request import urlopen

myparser = 'html.parser'
myurl = 'https://comic.naver.com/webtoon/weekday.nhn'
response = urlopen(myurl)
soup = BeautifulSoup(response, myparser)
print(type(soup))

# 요일별 폴더 생성
weekday_dict = {'mon':'월요일', 'tue':'화요일', 'wed':'수요일', 'thu':'목요일', 'fri':'금요일', 'sat':'토요일', 'sun':'일요일'}

# shutil : shell utility
import os, shutil
myfolder = 'C:\\Users\\bumsu\\SbaProject\\test01\\'

# 각 이미지를 저장해주는 함수
def saveFile(mysrc, myweekday, mytitle):
    image_file = urlopen(mysrc)
    filename = myfolder + weekday_dict[myweekday] + '\\' + mytitle + '.jpg'
    # print(mysrc)
    # print(filename)

    myfile = open(filename, mode='wb')
    myfile.write(image_file.read()) # 바이트 형태로 저장
    myfile.close()

try :
    if not os.path.exists(myfolder):
        os.mkdir(myfolder)

    for mydir in weekday_dict.values():
        mypath = myfolder + mydir

        if os.path.exists(mypath):
            # rmtree : remove tree
            shutil.rmtree(mypath)

        os.mkdir(mypath)

except FileExistsError as err :
    print(err)


mytarget = soup.find_all('div', attrs={'class':'thumb'})
print(len(mytarget))

mylist = [] # 데이터를 저장할 리스트

for abcd in mytarget:
    myhref = abcd.find('a').attrs['href']
    myhref = myhref.replace('/webtoon/list.nhn?', '')
    result = myhref.split('&')
    # print(myhref)
    # print(result)
    mytitleid = result[0].split('=')[1]
    myweekday = result[1].split('=')[1]
    # print(mytitleid)
    # print(myweekday)

    imgtag = abcd.find('img')
    mytitle = imgtag.attrs['title'].strip()
    mytitle = mytitle.replace('?', '').replace(':', '')
    # print(mytitle)

    mysrc = imgtag.attrs['src']
    # print(mysrc)

    saveFile(mysrc, myweekday, mytitle)
    # break

    sublist = []
    sublist.append(mytitleid)
    sublist.append(myweekday)
    sublist.append(mytitle)
    sublist.append(mysrc)
    mylist.append(sublist)

from pandas import DataFrame

mycolumns = ['타이틀번호', '요일', '제목', '링크']
myframe = DataFrame(mylist, columns=mycolumns)

filename = 'cartoon.csv'

myframe.to_csv(filename, encoding='utf-8', index=False)
print(filename + ' 파일로 저장됨')
print('finished')