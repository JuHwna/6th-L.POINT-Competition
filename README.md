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

## 5. 공모전 진행
### 1) 주어진 데이터 이해 및 사전 지식 획득
(1) 주어진 데이터 이해

- 총 4개의 데이터를 받았습니다.
  - 온라인 행동 정보(Behavior Table) : 업종별로 2019.07~2019.09까지 고객들의 온라인 행동 정보
    - CLNT_ID(클라이언트ID), SESS_ID(세션 ID), HIT_SEQ(조회일련번호), ACTION_TYPE(행동유형), BIZ_UNIT(업종단위), SESS_DT(세션일자), HIT_TM(조회시각), HIT_PSS_TM(조회경과시간), TRANS_ID(거래 ID), SECH_KWD(검색 키워드), TOT_PAG_VIEW_CT(총페이지조회건수), TOT_SESS_HR_V(총세션시간값), TRFC_SRC(유입채널), DVC_CTG_NM(기기유형)
  
  - 거래 정보(Transaction Table) : 온,오프라인 업종별로 2019.07~2019.09까지 고객들의 구매 정보
    - CLNT_ID(클라이언트ID), TRANS_ID(거래 ID), TRANS_SEQ(거래일련번호), BIZ_UNIT(업종단위), PD_C(상품소분류코드), DE_DT(구매일자), DE_TM(구매시각), BUY_AM(구매금액), BUY_CT(구매수량)
  - 고객 Demographic 정보 : 고객들의 id와 성별, 연령대 정보
    - CLNT_ID(클라이언트ID), CLNT_GENDER(성별), CLNT_AGE(연령대)
  - 상품 분류 정보 : 상품 정보(대분류,중분류,소분류 등)
    - PD_C(상품 소분류코드),CLAC1_NM(상품 대분류명),CLAC2_NM(상품 종분류명), CLAC3_NM(상품 소분류명)
- 먼저 각 데이터별로 특징들을 찾기 시작했습니다.
  - 온라인 행동 유형의 특징
     1) 업종별로 온라인 행동 유형이 다른 점을 발견했습니다.
        - A01은 0,6만 있지만 A02,A03은 0~7까지 다 있음을 발견했습니다.
           - A01만 왜 이런지 주관사에 문의를 했더니 계열사별로 얻을 수 있는 행동 유형이 다 달라서 그렇다는 답변만 받았습니다.
        - A02,A03의 유형 비율을 봤을 때 구매(6)의 비율이 매우 적음을 확인했습니다.
        - 또한 결제 시도 대비 구매 비율도 매우 작으므로 이를 활용해서 잠재고객을 사용하면 되겠다는 생각이 들었습니다.
           - 결제 시도까지 간 고객이 구매 의사가 있다는 점이 대부분 아는 사실이므로 그 점을 사용했습니다.
           - **저희 팀이 정한 잠재고객 정의** : **결제 시도를 했지만 구매하지 않은 사람들**
           
     ![image](https://user-images.githubusercontent.com/49123169/79116238-e9ed6c80-7dc2-11ea-9048-fcbcbba4c784.png)
   
     2) 기기 유형과 유입 유형을 분석한 결과, unknown 값이 있음을 발견했습니다.
        - unknown 값이 매우 많았지만 관련된 정보가 없어서 문의를 넣어 확인했습니다.
          - 문의한 결과, 저희가 가정한 모든 사례가 맞을 수도 있다는 답변만 받았습니다.
          - 그래서 정답이 없다는 생각에 **저희가 가정한 사실을 기반으로 해석**했습니다.
          
     ![image](https://user-images.githubusercontent.com/49123169/79116865-83694e00-7dc4-11ea-88cd-0c4c87ef0b1e.png)
     
     
     3) 자정이 지나면 조회경과시간, 총페이지 조회건수, 총세션시간값이 초기화됨을 확인했습니다.
     
     ![image](https://user-images.githubusercontent.com/49123169/79117036-f70b5b00-7dc4-11ea-9277-e6a00b1e5b7f.png)

     4) 이미 발생한 TRANS_ID가 일정 시간이 지난 뒤 다시 발생함을 확인했습니다.
        - 한 거래에 TRANS_ID가 한 번만 발생해야 되는데 그러지 않아서 팀원들끼리 많은 고민을 했습니다.
        - 고민한 결과, 이미 발생한 TRANS_ID는 **거래 내역 페이지를 확인**하는 것으로 보았습니다.
           - 페이지뷰수가 그것만 보고 나간 사람들이 많아 이렇게 가정했습니다.
     
     ![image](https://user-images.githubusercontent.com/49123169/79117231-89abfa00-7dc5-11ea-87eb-e770106a793e.png)
     
     5) 구매가 발생했는데 TRANS_ID가 NULL이면서 지속적으로 많이 발생함을 확인했습니다.
        - 거래를 했지만 TRANS_ID가 없으면서 구매 행동이 6인 경우가 보였습니다.
        - 고민한 끝에 이것 또한 **거래 내역 페이지 확인**으로 보았습니다.
        
     ![image](https://user-images.githubusercontent.com/49123169/79117360-e5768300-7dc5-11ea-9173-032fd59c4582.png)
        
   - 거래 정보 특징
     1) 각 계열사별 구매 주기를 구한 결과 대부분 7~14일 내에 재구매가 발생함을 알 수 있었습니다.
     
     ![image](https://user-images.githubusercontent.com/49123169/79122856-c41c9380-7dd3-11ea-9539-5cb490db7de6.png)

     2) 각 계열사별 시간에 따른 거래 건수가 다름을 알 수 있었습니다.
        - 오프라인에서 구매가 일어날 수 없는 시간에 구매가 있음을 확인했습니다.
        - 직원들이 산 건 아닐까라는 생각을 했지만 말이 새벽 3시에 구매하는 등 말도 안 되는 경우여서 이상치로 판단했습니다.
     
     3) 상품 금액이나 구매 갯수가 비이상적으로 높은 것들이 있음을 확인했습니다.
        - 이상치로 판단해 제거했습니다.
        
     4) 상품 금액이 0인 경우, 구매 갯수가 0인 경우, 상품 금액과 구매 갯수가 0인 경우가 있음을 확인했습니다.
        - 상품 금액이 0인 경우 : 여행자 보험과 같은 상품으로 생각했습니다.
        - 구매 갯수가 0인 경우 : 1+1 또는 사은품으로 판단했습니다.
        - 둘 다 0인 경우 : 행사용 제품으로 판단했습니다.
        
        
     
    - 고객 정보 특징
      1) 대다수의 고객은 여성임을 확인했으며 연령대는 30~40대가 많이 구매함을 알 수 있었습니다.
      
      
    - 상품 분류 정보 특징
      1) 대분류를 사용해서 각 계열사를 추정했습니다.(계열사를 알려주지 않아서 자체적으로 판단해서 추론했습니다.)
         - 결과적으로 온라인 한 곳을 제외하고 다 맞추었습니다!
         
    - 각 테이블을 같이 조인해서 봤을 때의 특징
      1) 온라인 행동 테이블과 거래 테이블을 조인했을 때 많은 trans_id가 누락됨을 확인했습니다.
         - 심각할 정도로 누락되어 원인에 대해 살펴본 결과 두 테이블의 구매 패턴이 상이함을 확인했습니다.
         - 이로 인해 두 테이블을 조인해서 사용하는 것보단 따로 보는 게 맞다고 판단했습니다.
         - 또한 분석을 진행할 때 온라인을 위주로 해야 되겠다는 결정을 내렸습니다.
         
      2) 온라인 행동 테이블과 거래 테이블의 trans_id가 같지만 거래시간이 불일치함을 발견했습니다.
         - 둘 중 더 빠른 시간대로 시간을 변경했습니다.
         
         
 - 테이블 특징을 살펴본 후, 저희 팀은 방향성을 구체적으로 잡았습니다.
    - A01만 행동 패턴이 다르기 때문에 고객 유형화할 때 A01에 맞는 방법으로, A02,A03 둘에 맞는 방법으로 진행하기로 했습니다.
    - 온라인 행동 테이블과 거래 테이블의 조인하고 사용하기가 불가능하기 때문에 온라인 행동 테이블에 집중하기로 결정했습니다.
      - 온라인 계열사 위주로 고객 유형화 및 추천 알고리즘 개발을 하기로 했습니다.
    - 
        
