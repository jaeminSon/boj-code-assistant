# 백준 코딩 도우미


### OpenAI API key 세팅
- authentication/openai.yaml 에 [api_key](https://platform.openai.com/docs/quickstart) 복붙

### 실행
```
$ python run.py
```

### 문제, 정답에서 힌트 얻기
- problems 폴더 밑에, 문제 텍스트를 <boj_문제번호>.txt 에 저장
- solutions 폴더 밑에, 정답 코드를 <boj_문제번호>.txt 에 저장 


### 문제 긁어 오기 (web-crawling 은 boj 에서 권장하지 않음)
```
$ cd data_preparation
$ python crawl_boj_problems.py
```

### 문제 meta 정보 다운받기
```
$ cd data_preparation
$ python collect_problem_list.py
```

### 알고리즘 설명 (재)생성
```
$ cd data_preparation
$ python collect_algorithm_description.py 
```
