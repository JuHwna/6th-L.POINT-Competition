# 제6회 L.POINT Big Data Competition
## 1. 주관 : L.POINT
## 2. 일정 : 2019.11.27 ~ 2020.2.18
## 3. 공모 주제 
- 주제 1 : Audience Targeting(부제 : 잠재고객 맞춤 컨텐츠 추천)
  - 세부과제
     1) 비식별 온라인 잠재고객의 행동/소비 패턴 생성
     2) AI 기반 고객 맞춤 상품 추천 알고리즘 도출
     3) 1,2를 활용한 비즈니스 전략 수립
  - **작성자**는 해당 주제로 참가했음을 알립니다
  
## 4. 공모전 참가 관련 사항
### 1) 팀 이름
- 개인이 아닌 팀으로 참가했습니다.
- 팀 이름 : **KUGGLE**

### 2) 팀장 및 팀원

|팀장 및 팀원|이름|
|-----------|----|
|팀장|노주영|
|팀원|박주환,박상희|

### 3) 각 구성원 역할

|역할|이름|
|----|----|
|EDA, 비즈니스 전략 수립, PPT 제작|노주영|
|EDA, 크롤링,고객 유형화 작업 및 분석, 비즈니스 전략 수립, PPT 제작, 발표|박상희|
|EDA, 크롤링, 추천 알고리즘 개발, 비즈니스 전략 수립, PPT 제작|박주환|

### 4) 사용 툴

|사용 툴|용도|
|-------|----|
|Oracle SQL|데이터를 이해하고 특징을 찾아내는 작업을 위해 사용했습니다.|
|Excel|EDA에 사용될 그래프를 더 명확하게 표현하고자 사용했습니다|
|Python|고객 유형화 작업, 크롤링, 추천 알고리즘 개발 등 공모전에 필요한 모든 분야를 수행하고자 사용했습니다.|
|Power Point|제출과 발표에 필요한 PPT를 제작하기 위해 사용했습니다|

## 5. 공모전 데이터
- 총 4개의 데이터를 받았습니다.
  - 온라인 행동 정보(Behavior Table) : 업종별로 2019.07~2019.09까지 고객들의 온라인 행동 정보
    - CLNT_ID(클라이언트ID), SESS_ID(세션 ID), HIT_SEQ(조회일련번호), ACTION_TYPE(행동유형), BIZ_UNIT(업종단위), SESS_DT(세션일자), HIT_TM(조회시각), HIT_PSS_TM(조회경과시간), TRANS_ID(거래 ID), SECH_KWD(검색 키워드), TOT_PAG_VIEW_CT(총페이지조회건수), TOT_SESS_HR_V(총세션시간값), TRFC_SRC(유입채널), DVC_CTG_NM(기기유형)
  
  - 거래 정보(Transaction Table) : 온,오프라인 업종별로 2019.07~2019.09까지 고객들의 구매 정보
    - CLNT_ID(클라이언트ID), TRANS_ID(거래 ID), TRANS_SEQ(거래일련번호), BIZ_UNIT(업종단위), PD_C(상품소분류코드), DE_DT(구매일자), DE_TM(구매시각), BUY_AM(구매금액), BUY_CT(구매수량)
  - 고객 Demographic 정보 : 고객들의 id와 성별, 연령대 정보
    - CLNT_ID(클라이언트ID), CLNT_GENDER(성별), CLNT_AGE(연령대)
  - 상품 분류 정보 : 상품 정보(대분류,중분류,소분류 등)
    - PD_C(상품 소분류코드),CLAC1_NM(상품 대분류명),CLAC2_NM(상품 종분류명), CLAC3_NM(상품 소분류명)
    
## 6. 공모전 진행 과정
**상세한 과정은 process 폴더 참고**
### 1) EDA 및 사전 조사
#### (1) EDA
- SQL과 파이썬을 통해 특정 조건으로 묶은 후, 숫자를 통해 이해 가능한 결과는 그대로 보았습니다.
- 숫자로 이해하기 어려울 경우, 파이썬의 matplotlib, seaborn, plotly를 사용하여 그래프를 그렸거나 Excel을 이용해 그래프로 나타내서 봤습니다.

#### (2) 사전 조사
- 고객 행동 유형화하기 위해 필요한 지식들을 습득하기 위해 고객 유형에 대해 논문을 읽었습니다.

##### 해당 과정 속의 결과
- EDA를 통해 총 세 가지의 방향성을 잡을 수 있었습니다.
   1) 온라인 행동 정보와 거래 정보를 조인할 시 거의 모든 데이터를 손실하기 때문에 온라인 행동 정보로만 분석을 진행
   2) 구매고객과 잠재고객의 기준을 5와 6으로 두기
      - 구매 고객 : 구매(6)까지 완료한 고객
      - 잠재 고객 : 결제시도(5)까지 했으나 구매까지 이어지지 않은 고객
   3) 이상치들을 가정하거나 제거한 후 분석 진행
- 또한 사전 조사를 통해 고객 유형화 작업의 기준을 세웠습니다.
   - 기준 : **구매까지 걸리는 고민기간** 사용
   
### 2) 고객 유형화 작업
- 고민기간이 상품의 금액에 따라 달라지기 때문에 각 계열사마다 다르게 고민기간을 부여했습니다.
- 그 후, 구매에 영항을 

