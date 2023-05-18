### 題目：波士頓房價預測

### 數據摘要
|  名稱   | 波士頓房價數據資料  |
|  ----  | ----  |
| 特徵簡介  | CRIM：人均城鎮犯罪率<br>ZN：劃為25,000平方英尺以上土地的住宅用地比例<br>INDUS：每個城鎮非零售業務英畝的比例<br>CHAS：查爾斯河虛擬變量<br>NOX：一氧化氮濃度(百萬分之一)<br>RM：每個住宅的平均房間數<br>AGE：1940年之前建造的自有住房的比例<br>DUS：到五個波士頓就業中心的加權距離<br>RAD：徑向公路的可達性指數<br>TAX：每10,000美元的全職財產稅率<br>PTRATIO:城鎮的師生比例<br>B：1000（Bk - 0.63）^2其中Bk是黑人比例<br>LSTAT：中下階級的比率<br>Target：城鎮房價的中間價格|
| 資料數  | 506筆 |
| 分析目標  | 建立一個已知特徵並預測波士頓未知房價的模型 |
| 分析分法  | 透過線性迴歸的方式來進行預測 |

## 1. 簡介

### 1.1 資料簡介

#### 1.1.1 資料來源

本次練習是採用sklearn所提供的波士頓房價資料

#### 1.1.2 特徵簡介

CRIM：人均城鎮犯罪率<br>ZN：劃為25,000平方英尺以上土地的住宅用地比例<br>INDUS：每個城鎮非零售業務英畝的比例<br>CHAS：查爾斯河虛擬變量<br>NOX：一氧化氮濃度(百萬分之一)<br>RM：每個住宅的平均房間數<br>AGE：1940年之前建造的自有住房的比例<br>DUS：到五個波士頓就業中心的加權距離<br>RAD：徑向公路的可達性指數<br>TAX：每10,000美元的全職財產稅率<br>PTRATIO:城鎮的師生比例<br>B：1000（Bk - 0.63）^2其中Bk是黑人比例<br>LSTAT：中下階級的比率<br>Target：城鎮房價的中間價格
#### 1.1.3 資料的統計特徵
![.png](https://upload.cc/i1/2023/05/16/rKRbNI.png)

### 1.2 分析工具和方法

#### 1.2.1 分析工具
&emsp;&emsp;線性迴歸是一種常用的統計分析方法，用於建立自變數和應變數之間的線性關係模型。透過迴歸分析，我們可以推斷自變數對應變數的影響程度，並用這種關係來預測未知的應變數數值。

#### 1.2.2 分析方法
利用sklearn，建立線性迴歸模型進行波士頓房價預測，利用R-square、MAE、MSE及RMSE來衡量模型的表現。

## 2. 資料處理過程

### 2.1 資料匯入與基本處理

#### 2.1.1 導入資料
&emsp;&emsp;利用sklearn匯入波士頓房價資料
~~~python
    from sklearn.datasets import load_boston
    boston = load_boston()
~~~

#### 2.1.2 資料整理
&emsp;&emsp;將原始數據和Target合併成DateFrame
~~~python
    x = boston.data
    y = boston.target
    df = pd.DataFrame(x, columns=boston.feature_names)
    df['Target'] = pd.DataFrame(y, columns=['Target'])
~~~

#### 2.1.3 確認是否有空值
&emsp;&emsp;
~~~python
    df.info()
~~~
![.png](https://upload.cc/i1/2023/05/16/T5nFdt.png)

#### 2.2 資料分析圖表繪製

#### 2.2.1 熱力圖及相關矩陣
~~~python
    plt.figure(figsize=(15,10))
    plt.title('Correlation between features')
    sns.heatmap(df.corr(), annot=True, square=True)
    plt.show()
~~~
![.png](https://upload.cc/i1/2023/05/16/O4ayYn.png)

#### 2.2.2 房價直方圖以及Q-Q plot
~~~python
    sns.distplot(df["Target"])
    plt.show()
    stats.probplot(df['Target'], plot=pylab)
    pylab.show()
~~~
![.png](https://upload.cc/i1/2023/05/17/4WERrH.png)
![.png](https://upload.cc/i1/2023/05/17/4WERrH.png)

### 2.3 波士頓房價預測

#### 2.3.1 線性迴歸

切割資料及建立模型
~~~python
    # 匯入模組
    from sklearn.linear_model import LinearRegression 
    from sklearn.model_selection import train_test_split 
    # 切割資料
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 5) # 將數據分成73比
    # 創建模型
    model = LinearRegression()
    model.fit(x_train, y_train) # 將資料拿去訓練
~~~

實際值與預測值可視化對比
~~~python
    pred = model.predict(x_test)
    # 印出結果
    pd.DataFrame({
        'actual_y': y_test,
     'pred_y': pred
    })
# 預測結果可視化
    fig = plt.figure(figsize=(12,6))
    y_test_index = np.arange(y_test.shape[0])
    pred_index = np.arange(pred.shape[0])
    plt.plot(y_test_index, y_test, color='black', linestyle='-', linewidth=1.5)
    plt.plot(pred_index, pred, color='red', linestyle='-.', linewidth=1.5)
    plt.xlim((0,152))
    plt.ylim((0,55))
    plt.legend(['actual','predict'])
    plt.show()
~~~
![.png](https://upload.cc/i1/2023/05/17/UtPYoF.png)
![.png](https://upload.cc/i1/2023/05/17/taGew3.png)

線性迴歸模型的評估
~~~python
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    # 印出預測績效(R-square)
    print(f'Linear Regression\'s score: {model.score(x_test, y_test)}')
    # 印出其他迴歸績效指標

    pd.DataFrame({
        'R-square': [r2_score(y_test, pred)],
        'MAE': [mean_absolute_error(y_test, pred)],
        'MSE': [mean_squared_error(y_test, pred)],
        'RMSE': [mean_squared_error(y_test, pred, squared=False)]
    },index=['value'])
~~~
![.png](https://upload.cc/i1/2023/05/17/NlokCD.png)

## 3. 結果分析

### 3.1 分析各特徵與房價的相關性
1.人均城鎮犯罪率(CRIM)越低，房價越高。<br>
2.劃為25,000平方英尺以上土地的住宅用地比例(ZN)與房價相關性較低。<br>
3.每個城鎮非零售業務英畝的比例(INDUS)越高，房價越低，但INDUS較低時，房價也有低的。<br>
4.查爾斯河虛擬變量(CHAS)在附近的房價有高有低。<br>
5.一氧化氮濃度(NOX)相關性較低。<br>
6.每個住宅的平均房間數(RM)房間數越多，房價越高。<br>
7.1940年之前建造的自有住房的比例(AGE)與房價相關性較低。<br>
8.到五個波士頓就業中心的加權距離(DUS)距離就業中心越遠，房價就越高。<br>
9.徑向公路的可達性指數(RAD)與房價沒有任何相關性。<br>
10.每10,000美元的全職財產稅率(TAX)稅率高，房價有高也有低。<br>
11.城鎮的師生比例(PTRATIO)越高，房價越低的數目越多。<br>
12.B：1000（Bk - 0.63）^2其中Bk是黑人比例都落在400，房價有高有低。<br>
13.中下階級的比率(LSTAT)越低，房價越低。<br>

### 3.2 預測結果分析
&emsp;&emsp;線性迴歸的(R-square)為0.6772，MAE為3.5577，MSE為30.697，RMSE為5.5404，有67.72%可以被自變數以線性迴歸模型來解釋，但仍有32.28%的變異未被模型解釋。較高的判定係數表示數據與模型有高度擬合，













