## Research Question 

Leaving behind the briefly calming spread of COVID-19, the world is doing its best to minimize the spread of coronavirus as COVID-19 is actively spreading again. To minimize the spread, we need to identify and deal with the exact factors. There are also cases in which Kaggle has published a challenge, such as COVID19 Global Forecasting (Week 1). What I noticed was the number of confirmed cases varies from country to country. For sure, there can be various factors that can vary from country to country, but for example, response policies by country, people's tendencies, and the trust in the government. I thought that the personality of country might be related to the spread of the coronavirus. Therefore, use K-means clustering to identify the relationship between the number of confirmed cases of corona in countries with the personality of country.

## Data
1. [COVID-19](https://covid19.who.int/table)
The coronavirus data was obtaiend by the WHO Coronavirus Disease (COVID-19) Dashboard to get state-of-the-art coronavirus data.

2. [Big Five Personality Test](https://www.kaggle.com/tunguz/big-five-personality-test)
The Big Five personality test in the cavalier was used to identify national personalities. Korea, for example, is as follows. In the Big Five personality test in the Kaggle, data of people of Korean nationality were collected, averaged, and analyzed in terms of Korea's tendency. A quick look at the Big Five personality test is basically in the form of answering a questionnaire, with 5 points for agreeing completely with each question rating from 1 to 5 points, and 1 point for vice versa. The questions consist of five questions that can be identified from each of the Big Five personality tests. As a result, the personality can be identified by the sum of the answers to the questionnaire. The five personalities are classified as neuroticism, extraversion, O (Openness to Experience), Agreeableness, and Conscientiousness (C).

## Clustering
To set the K-means clustering model, the hyper parameter, k, must be set. Two methods were used to set the k. First, through the **elbow plot**, the appropriate k was identified to narrow the range. *However, It's hard to choose k directly and sorely by the elbow plot.* If the number of contries in one cluster is small, then I considered that it's not representative to the cluster. By applying it to the model with values of narrow k, we check whether the distribution for each cluster is even to determine the final k . As a result of all these processes, k is set to 2.
```py
# elbow plot
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_train.iloc[:,3:8])
    distortions.append(kmeanModel.inertia_)
   

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distance')
plt.title('The Elbow Plot')
plt.vlines(x=3, ymin=0, ymax=300, alpha=0.5, color='blue', linestyle='dashed', linewidth=1.0)
plt.show()
```
![Elbow](https://user-images.githubusercontent.com/70493869/95648294-133c4a80-0b11-11eb-9bb0-53f1a1ee7e6e.png)

```py
# k에 따른 분포 비교

for k in range(2,5):
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_train.iloc[:,3:8])
    df_train['군집'] = kmeanModel.labels_
    print('k = ', k)
    print(df_train['군집'].value_counts(),'\n')
```
```
k =  2
0    26
1    24
Name: 군집, dtype: int64 
```
```
k =  3
1    22
2    16
0    12
Name: 군집, dtype: int64 
```
```
k =  4
0    22
2    13
1     9
3     6
Name: 군집, dtype: int64 
```

```py
#k = 2로 설정
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_train.iloc[:,3:8])
df_train['군집'] = kmeans.labels_
```

   국가 | 국가코드 | 확진자 수 | 개방성 | 성실성 | 외향성 | 우호성 | 안정성 | 군집
| -- | -- | -- | -- | -- | -- | -- | -- | --
Argentina | AR | 798486 | 40.133075 | 31.678530 | 28.345068 | 36.416248 | 32.381431 | 1
Australia | AU | 27149 | 38.551472 | 33.626061 | 29.937915 | 37.918328 | 30.393272 | 0
Austria | AT | 49886 | 40.836251 | 32.518538 | 29.899588 | 36.876931 | 29.856334 | 0
Belgium | BE | 132109 | 39.308450 | 32.029262 | 29.967453 | 37.661989 | 30.598089 | 0
Brazil | BR | 4915289 | 40.536946 | 31.240467 | 26.505017 | 34.946725 | 31.545703 | 1

Here, because the number of confirmed cases in the United States and India is relatively large compared to other countries, to produce more general results, the two countries were considered as ouliers. 

```py
# 인도와 미국의 확진자수를 Outlier 취급

i = df_train.loc[(df_train['국가'] == 'India') | (df_train['국가'] == 'United States of America')].index

df_train.drop(i, inplace=True)
```

## Result
### 1. Train
```py
# 군집별 분류

traits = ['개방성','신경성','외향성','우호성','안정성']

train_res = df_train.iloc[:, 2:]\
                    .groupby('군집')\
                    .mean()\
                    .rename_axis('군집')\
                    .reset_index()
train_res = train_res.apply(lambda x: x.round(2))
train_res = train_res.sort_values('확진자 수', ascending=False).reset_index(drop=True)
```

   군집 | 확진자 수 | 개방성 | 성실성 | 외향성 | 우호성 | 안정성
| -- | -- | -- | -- | -- | -- | --
1 | 512379.08 | 39.65 | 32.13 | 28.48 | 35.84 | 31.62
0 | 168060.58 | 38.62 | 33.39 | 29.80 | 37.43 | 30.49

### 2. Test
```py
# test 나라들의 데이터 예측
mean_test['군집'] = kmeans.predict(mean_test.iloc[:,1:])
```

   국가코드 | 개방성 | 성실성 | 외향성 | 우호성 | 안정성 | 군집
| -- | -- | -- | -- | -- | -- | --
CR | 39.633028 | 32.664220 | 28.759633 | 35.908257 | 31.051376 | 1
EC | 38.383138 | 31.910352 | 29.119530 | 35.337247 | 31.806830 | 1
EE | 38.964045 | 32.362921 | 28.283146 | 35.783146 | 30.279775 | 1
EG | 37.943890 | 33.007481 | 28.480050 | 36.966334 | 33.011222 | 1
IS | 39.057728 | 32.614525 | 30.225326 | 37.696462 | 30.463687 | 0
```py
# 군집별 분류
df_test = mean_test.merge(df_covid, on = '국가코드')


test_res = df_test.loc[:, ['군집', '확진자 수'] + traits]\
                    .groupby('군집')\
                    .mean()\
                    .rename_axis('군집')\
                    .reset_index()
test_res = test_res.apply(lambda x: x.round(2))
test_res = test_res.sort_values('확진자 수', ascending=False).reset_index(drop=True)
```

   군집 | 확진자 수 | 개방성 | 성실성 | 외향성 | 우호성 | 안정성
| -- | -- | -- | -- | -- | -- | --
1 | 100305.4 | 39.24 | 32.40 | 28.40 | 35.64 | 31.53
0 | 32628.0 | 38.57 | 34.19 | 29.64 | 37.93 | 30.71

First, we found that the number of confirmed cases was high on average in Cluster 1. And the tendency of Cluster 1 shows that openness is about one point higher than Cluster 0. On the other hand, friendliness was found to be relatively low compared to cluster 0. In short, openness can be seen as a preference for outdoor activities, which can be expected to result in a high number of confirmed cases in Cluster 1. On the other hand, Cluster 0 which is more open-minded but friendly shows better adaptation to a given environment, although it likes outdoor activities, but enjoys outdoor activities in other ways, suggesting that the average number of confirmed cases is lower than Cluster 1.
## Visualization 

![6](https://user-images.githubusercontent.com/70493869/95484620-a9feef00-09cb-11eb-8e57-d11a6d4151e2.png)

From the graph above, the clustering results of test data and train data could be seen. As expected, we could confirm that Cluster 0 has more confirmed cases than Cluster 1, and the most surprising fact is that both train and test data show that Cluster 0 has a similar ratio of confirmed cases. Of course, one of the factors behind the spread of Corona cannot be said to be the Big Five personality test. All these results may also have been lucky to come out like that. However, through this project, I think the personality of countries can be an indirect factor in the spread of the coronavirus.
