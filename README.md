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

#### 추천 모델 중간 과정
- 처음에 추천 모델 1으로 돌렸을 때 안 좋은 결과가 나왔습니다.
  - 특히 예측 결과가 하나의 검색어로 몰리는 결과가 많았습니다.
  - 각 검색어 비율을 검색한 결과 원피스가 압도적으로 많았다는 사실을 알 수 있었습니다.
- 그래서 검색어 대신 다른 요소를 집어 넣는게 좋을 거라는 생각이 들었습니다.
  - 분석 결과도 안 좋게 나올 뿐더러 검색어가 특정 상품군을 설명하기가 애매했기 때문입니다.
  - ex)원피스를 검색했을 때, 이 고객이 어떤 원피스를 원하는지를 모른다.
  
  ![image](https://user-images.githubusercontent.com/49123169/79782318-d53c4600-8379-11ea-9e40-5fb5b512dfcb.png)

  
- 검색어 대신 검색어를 검색했을 때 나오는 브랜드와 타입을 넣기로 했습니다.
  - 브랜드와 타입이 구체적이기 때문에 이 방법을 채택했습니다.
  - 해당 데이터는 각 계열사에 검색어를 입력한 후 나오는 결과를 수집했습니다.
  
  ~~~
  for i in range(len(unia02_kwd)):
    
    driver.implicitly_wait(randint(2,6))
    driver.get('http://www.lotteimall.com/main/viewMain.lotte?dpml_no=1&tlog=00100_1')
    a_element1=driver.find_element_by_id('headerQuery')
    a_element1.click()
    a_element1.send_keys(unia02_kwd[i])
    a_element2=driver.find_element_by_id("btn_headerSearch")
    a_element2.click()
    
    if ('결과' in driver.find_element_by_class_name('center').text) == True:

        a_element3=driver.find_element_by_xpath('//*[@id="contents"]/fieldset[2]/div/div[2]/ul/li[4]/a')
        a_element3.click()
        a_element4=driver.find_elements_by_xpath("//p[@class='title']")
        a_ss=[]
        for w in range(len(a_element4)):
            m = re.sub('r([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)','',a_element4[w].text)
            m=m.replace('[','')
            m=m.replace(']','')
            a_ss.append(m)
        a_ss=[x for x in a_ss if x]
        a_s2=[j.split(' ') for j in a_ss]
        a_s2=sum(a_s2,[])
        a_element6=[[x,a_s2.count(x)] for x in set(a_s2)]
        a_element6=pd.DataFrame(a_element6,columns=[i,'count'])
        a_element6=a_element6.sort_values('count',ascending=False).iloc[:5,:].values.tolist()
        a_element6=sum(a_element6,[])
        a_element6=pd.Series(a_element6,name=i)
        a02_brand2=pd.concat([a02_brand2,a_element6],axis=1)
    else:
        a_element6=pd.Series(['검색이 되지 않는다'])
        a_element6=a_element6.rename(i)
        a02_brand2=pd.concat([a02_brand2,a_element6],axis=1)
    
        
    print(i)
  ~~~


#### 추천 모델 1. FM모델(Factorization machine)
##### 선택 이유 
  - 잘 알려진 추천 모델의 경우, 고객이 아이템에게 준 평점만을 사용해서 추천 시스템을 구현합니다.
  - 하지만 저희가 가진 데이터는 평점 데이터도 없을 뿐더러 고객 정보 등 다양한 데이터가 있기 때문에 데이터 손실이 매우 커서 다른 모델을 찾았습니다.
  - >Factorization Machines (Steffen Rendle)
    - 해당 모델의 경우, 회귀분석처럼 여러 변수들을 넣고 돌려도 되는 추천 시스템 알고리즘인 것을 확인했습니다.
    
    ![image](https://user-images.githubusercontent.com/49123169/79763963-b630ba80-835f-11ea-9b8f-5a507b2a84d4.png)
    
    - 고객 행동 유형화, 고객 정보, 머무른 시간 등 주어진 데이터를 살릴 수 있어 해당 모델을 채택하여 모델을 돌릴 작업을 진행했습니다.

##### 데이터 전처리
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
  
##### 모델 학습 및 결과
- **첫 모델 학습**
  - 모든 변수들을 더미화 시킨 후에 진행했습니다.
    - 더 많은 변수들을 넣으려고 진행했지만 행이 매우 크지 않아서 열의 갯수를 조절하여 학습을 진행했습니다.
    - Y 값을 특정 지을 수 없었기 때문에 총 3가지 경우로 나누어서 모델을 학습시켰습니다.
    
    ![image](https://user-images.githubusercontent.com/49123169/79765165-3c99cc00-8361-11ea-9aeb-596c585bba81.png)

    
  - 학습 코드(tensorflow 1.xx 버전으로 진행)
  ~~~
  k = 5
  n, p = X_train.shape
  X = tf.placeholder('float', shape=[n, p])
  y = tf.placeholder('float', shape=[n, 1])

  w0 = tf.Variable(tf.zeros([1]))
  W = tf.Variable(tf.zeros([p]))
  V = tf.Variable(tf.random_normal([k, p], stddev=0.01))
  y_hat = tf.Variable(tf.zeros([n, 1]))
  linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
  pair_interactions = (tf.multiply(0.5,
                      tf.reduce_sum(
                          tf.subtract(
                              tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                              tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                          1, keep_dims=True)))
  y_hat = tf.add(linear_terms, pair_interactions)
  # L2 regularized sum of squares loss function over W and V
  lambda_w = tf.constant(0.001, name='lambda_w')
  lambda_v = tf.constant(0.001, name='lambda_v')

  l2_norm = (tf.reduce_sum(
              tf.add(
                  tf.multiply(lambda_w, tf.pow(W, 2)),
                  tf.multiply(lambda_v, tf.pow(V, 2)))))

  error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
  loss = tf.add(error, l2_norm)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
  
  N_EPOCHS = 100
  # Launch the graph.
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init)

      for epoch in range(N_EPOCHS):
          indices = np.arange(n)
          np.random.shuffle(indices)
          X_train, y_train = X_train[indices], y_train[indices]
          sess.run(optimizer, feed_dict={X: X_train, y: y_train})
          if epoch%10==0:
            print('MSE: ', sess.run(error, feed_dict={X: X_train, y: y_train}))

      print('MSE: ', sess.run(error, feed_dict={X: X_train, y: y_data}))
      print('Predictions:', sess.run(y_hat, feed_dict={X: X_train, y: y_train}))
      print('Learnt weights:', sess.run(W, feed_dict={X: X_train, y: y_train}))
      print('Learnt factors:', sess.run(V, feed_dict={X: X_train, y: y_train}))
  ~~~
  
  - 학습 결과, 처참한 결과를 볼 수 있었습니다.
  
  ![image](https://user-images.githubusercontent.com/49123169/79765567-bfbb2200-8361-11ea-92d6-d86bb2359c40.png)
  
  - 3가지 경우 모두 안 좋은 결과가 나와서 다른 모델을 써야할 상황이었습니다.
- **두번째 모델**
  - 다른 모델을 찾던 도중, FM 모델을 응용해보면 어떨까라는 생각이 들었습니다.
    - FM이 회귀모델과 비슷하다는 것을 이용해서 softmax 모델처럼 돌리면 어떨까라는 생각이 들었습니다.
    - 그래서 똑같은 전처리에다가 이전 검색 기록을 넣어서 돌려보았습니다.
    
     
    ![image](https://user-images.githubusercontent.com/49123169/79768245-7a98ef00-8365-11ea-8a70-6ba2e0b23b03.png)
    
  - 학습 코드(keras로 구현)
  ~~~
  his_list2=[]
  input_dim=len(x_train.columns)
  model2 = Sequential()
  model2.add(Dense(64, input_dim=input_dim,activation='relu'))
  model2.add(Dropout(0.25))
  model2.add(Dense(128,activation='relu'))
  model2.add(Dropout(0.25))
  model2.add(Dense(256,activation='relu'))
  model2.add(Dropout(0.25))
  model2.add(Dense(512,activation='relu'))
  model2.add(Dropout(0.25))
  model2.add(Dense(1024,activation='relu'))
  model2.add(Dropout(0.25))
  model2.add(Dense(703, activation='softmax'))
  model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history2=model2.fit(x_train, y_train, batch_size=64,epochs=200,verbose=2,validation_split=0.2)
  his_list2.append(history2)
  ~~~
  
  - 학습 결과
    - 생각보다 좋은 결과가 나와서 해당 모델을 채택해서 예선에 제출했습니다.
   
     ![image](https://user-images.githubusercontent.com/49123169/79768365-a6b47000-8365-11ea-846d-6ac6ee3e38b0.png)
    
    
    - 나중에 알고 봤더니 저희 팀이 원했던 한 행동 내에서 검색한 검색어를 모두 활용해서 상품 추천을 해주어야 하는데 이전 검색어로만 추천을 해주는 방식이었습니다. 즉, 고객이 3개를 검색했는데 마지막에 검색한 것만 가지고 상품을 추천해주는 방식이었습니다.
    
    - 그래서 본선에 진출한 이후에 다시 다른 모델을 찾기 시작했습니다.
    
#### 추천 모델 2. Wide & Deep Learning Model
- 모델 선정 이유
  - 고객의 하나의 행동 패턴에서 
