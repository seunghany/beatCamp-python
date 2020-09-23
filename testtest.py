# bs4Exam03.py

'''
수집할 내용
순위 : 숫자 2자리 형식으로 가져오기 (zfill() 함수)
제목
변동 : 불변, 상승, 하락
변동 값 : 숫자

크롤링 순서
<div class='tit3'> 인 항목찾기

'''

import  urllib.request

from  bs4 import BeautifulSoup
from  pandas import DataFrame

url = "http://movie.naver.com/movie/sdb/rank/rmovie.nhn"
html = urllib.request.urlopen(url)
soup = BeautifulSoup(html, 'html.parser')
print(type(soup))

tags = soup.findAll('div', attrs ={'class':'tit3'})
print(tags)
print('-'* 30)

for tag in tags :
    print(tag.a.string)

print('-'*30)

print('<a> 태그의 href 전체 태그')
url_header = 'https://movie.naver.com'
for tag in tags:
    print(url_header + tag.a['href'])

print('-' * 30)

mytrs = soup.find_all('tr')
print(len(mytrs))

no = 0  # 순서
totallist = [] # 전체를 저장할 리스트

for one_tr in mytrs :
    #print(one_tr)
    #print('@'*30)

    title = ''
    up_down = ''

    mytd = one_tr.find('td', attrs={'class':'title'})
    if(mytd != None):
        no += 1
        newno = str(no).zfill(2)

        mytag = mytd.find('div', attrs={'class':'tit3'})

        # string 속상 : 해당 태그가 가지고 있는 문자열 출
        title = mytag.a.string

        # td 태그 중에서 3번째 요소를 찾기
        mytd = one_tr.select_one('td:nth-of-type(3)')
        myimg = mytd.find('img')
        if myimg.attrs['alt'] == 'up':
            up_down = '상승'
        elif myimg.attrs['alt'] == 'down':
            up_down = '하락'
        else:
            up_down = '불변'

        change = one_tr.find('td', attrs={'class':'range ac'})
        change = change.string

        #print(newno +'/'+ title +'/' +up_down+'/' + change)
        totallist.append((newno, title, up_down, change))


mycolumn = ['순위', '제목', '변동', '변동값']
myframe = DataFrame(totallist, columns=mycolumn)

filename = 'naverMovie.csv'

myframe.to_csv(filename)

print(filename + ' 파일 저장됨')
print('-'*30)