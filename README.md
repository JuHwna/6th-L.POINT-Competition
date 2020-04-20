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

~~~
#파이썬으로 구현했을 때, 하나의 코드
plt.figure()
plt.rcParams["figure.figsize"] = (45,45)
plt.rc('font',size=20)
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=False, sharey=False)
ax1.bar(graph[graph['biz_unit']=='A01'].iloc[:,1],graph[graph['biz_unit']=='A01'].iloc[:,2])
ax1.set_title('A01',fontsize=50)
ax2.bar(graph[graph['biz_unit']=='A02'].iloc[:,1],graph[graph['biz_unit']=='A02'].iloc[:,2])
ax2.set_title('A02',fontsize=50)
ax3.bar(graph[graph['biz_unit']=='A03'].iloc[:,1],graph[graph['biz_unit']=='A03'].iloc[:,2])
ax3.set_title('A03',fontsize=50)
ax4.bar(graph[graph['biz_unit']=='B01'].iloc[:,1],graph[graph['biz_unit']=='B01'].iloc[:,2])
ax4.set_title('B01',fontsize=50)
ax5.bar(graph[graph['biz_unit']=='B02'].iloc[:,1],graph[graph['biz_unit']=='B02'].iloc[:,2])
ax5.set_title('B02',fontsize=50)
ax6.bar(graph[graph['biz_unit']=='B03'].iloc[:,1],graph[graph['biz_unit']=='B03'].iloc[:,2])
ax6.set_title('B03',fontsize=50)
~~~

#### (2) 사전 조사
- 고객 행동 유형화하기 위해 필요한 지식들을 습득하기 위해 고객 유형에 대해 논문이나 관련 사이트를 찾아 글을 읽었습니다.
  >Designing for 5 Types of E-Commerce Shoppers "Nielsen Norman Group"
  
![image](https://user-images.githubusercontent.com/49123169/79756433-91374a00-8355-11ea-8f7d-17f4ebc5fe83.png)


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
- 고민기간을 사용하기로 했지만 어떤 행동을 기준으로 유형화 작업을 할지 의논을 나누었습니다.
  - 구매고객의 경우 : 구매(6)을 기준으로 행동 유형 나누기
  - 잠재고객의 경우 : 결제시도(5)를 기준으로 행동 유형 나누기
- 해당 기준으로 나눌 것이기 때문에 그 이외의 행동 유형(0,1,2,3,4)는 0으로 바꾸기로 했습니다.
- 그 후, 구매에 영항을 미치는 요인들까지 생각해서 고객 행동에 따라 유형을 나누는 작업을 진행했습니다.
  - A : 기준 action_Type에 영향을 끼칠 수 있는 기간 -> 각 계열사별 고민기간
  - B : 기준 action_type간 영향을 끼칠 수 있는 기간(재구매를 하는데 있어 이전 구매가 영향을 끼칠 수 있는 일자) -> 1일
  - C : 기준 action_type 이후의 행동에 영향을 끼칠 수 있는 기간 -> 1일
- 해당 기준으로 유형화한 결과, 총 1만개 이상의 행동 유형이 나타남을 확인할 수 있었습니다.
  - 하지만 0,5/0,6이 반복하는 패턴이 매우 많이 나타나 한번 더 유형화 작업으로 분석 시, 유용하게 사용할 수 있을 거라고 판단했습니다.
  - 그래서 각 계열사 별로 한 번 더 고객 행동을 유형화 했습니다.
    - A01 : category1(6), category2(0,6), category3((0,6)<sup>2</sup>~(0,6)<sup>4</sup>), 그 외
    - A02, A03 : category1(0,5,6), category2((0,5)<sup>2</sup>~ (0,5)<sup>4</sup>,6), category3((0,5)<sup>5</sup>~(0,5)<sup>9</sup>,6), 그 외
    
    ![image](https://user-images.githubusercontent.com/49123169/79756459-9b594880-8355-11ea-8fa3-1e5f4821f113.png)
    
    
#### 유형화 후, 분석 결과
- 구매 고객의 특징
   1) 모든 계열사에서 구매한 고객들은 대부분 누적 이용시간 1시간 이내에 구매한 것을 확인했습니다.
      - 이 특징을 기반으로 잠재고객이 1시간 이내에 구매할 확률이 높을 것으로 추정
   2) 시간대별 구매건수가 계열사마다 다르다는 것을 볼 수 있었습니다.
      - A01과 A02는 오전 10시에 증가하여 22~23시에 최대치를 기록
      - A03은 오전 10시에 증가하여 11시에 최대치를 기록
      
   ![image](https://user-images.githubusercontent.com/49123169/79756739-09057480-8356-11ea-9a71-fd84fc5a08de.png)

 
      
- 잠재고객 분석의 특징
   1) 결제시도는 했지만 구매하지 않은 고객을 발견했습니다.
      - 온라인 행동 테이블에 고객에 대한 정보가 있지만 고객테이블에 없는 경우를 발견 : **비회원고객**이라고 가정
         - 롯데 쇼핑몰의 개인정보처리방침 약관에 따라 가정했습니다.
   2) 구매했지만 고객정보가 없는 고객
      - 구매했지만 고객정보가 없는 고객을 비회원고객이라고 판단했습니다.
      - 회원과 비회원의 구매력을 비교한 결과, 회원의 구매력이 비회원의 구매력보다 높았습니다.
   3) 결제시도를 반복하고 구매하지 않은 고객
      - 5회 이상 연속으로 결제시도를 한 고객이 약 10%가 있는 것을 확인했습니다.
      - 결제시도 오류 또는 어려움으로 구매완료에 실패한 사례로 분석했습니다.
      
![image](https://user-images.githubusercontent.com/49123169/79756768-16bafa00-8356-11ea-903c-34907f3bcfb8.png)  

### 3) 추천 모델 개발
- 추천 모델 타겟팅
  - 공모전 주제가 잠재고객 맞춤 컨텐츠 추천이기에 잠재고객을 타겟으로 한 추천 모델을 만들고자 했습니다.
  - 그런데 공모전에서 나눠 준 자료에 따르면 상품 추천 알고리즘을 만들어달라고 했습니다.
  - 그래서 저희가 만든 데이터를 통해 어떻게 추천 모델을 만들지 고민했습니다.
    - 결과 : 잠재고객을 위한 상품 추천 모델 개발(1시간 이내 구매 유도)
- 어떤 방법으로 잠재고객에게 추천을 할 것인가?
  - 구매 고객의 구매 내역을 통해 비슷한 행동 유형을 한 고객에게 추천해주는 것으로 정했습니다.
  - 그래서 구매 고객의 온라인 행동 테이블을 주 분석 데이터로 사용했습니다.


#### 추천 모델 1. FM모델(Factorization machine)
- 선택 이유 
  - 잘 알려진 추천 모델의 경우, 고객이 아이템에게 준 평점만을 사용해서 추천 시스템을 구현합니다.
  - 하지만 저희가 가진 데이터는 평점 데이터도 없을 뿐더러 고객 정보 등 다양한 데이터가 있기 때문에 데이터 손실이 매우 커서 다른 모델을 찾았습니다.
  - >Factorization Machines (Steffen Rendle)
    - 해당 모델의 경우, 회귀분석처럼 여러 변수들을 넣고 돌려도 되는 추천 시스템 알고리즘인 것을 확인했습니다.
    
    ![image](https://user-images.githubusercontent.com/49123169/79763963-b630ba80-835f-11ea-9b8f-5a507b2a84d4.png)
    
    - 고객 행동 유형화, 고객 정보, 머무른 시간 등 주어진 데이터를 살릴 수 있어 해당 모델을 채택하여 모델을 돌릴 작업을 진행했습니다.

- 데이터 전처리
  - 해당 논문에 있는 것처럼 데이터를 만들기 위해 여러 작업들을 거쳤습니다.
  
  ~~~
  dayofweek_1_onehot=pd.get_dummies(new_category_1['dayofweek'])
  hour_1_onehot=pd.get_dummies(new_category_1['hour'])
  test2=pd.concat([cate_1_onehot,kwd_1_onehot,other_item,dayofweek_1_onehot,hour_1_onehot],axis=1)
  target=new_category_1['target']
  test5=test2.values
  target=target.values
  X_train, X_test, y_train, y_test = train_test_split(test5, target, test_size=0.2)
  ~~~
  
  - 모든 변수들을 더미화 시킨 후에 진행했습니다.
    - 더 많은 변수들을 넣으려고 진행했지만 행이 매우 크지 않아서 열의 갯수를 조절하여 학습을 진행했습니다.
    
  - 학습 코드(tensorflow 1.xx 버전으로 진행)
  ~~~
  
  ~~~
    
