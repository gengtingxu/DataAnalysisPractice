### 題目：波士頓房價預測

### 數據摘要
|  名稱   | 波士頓房價數據資料  |
|  ----  | ----  |
| 特徵簡介  | CRIM：人均城鎮犯罪率<br>ZN：劃為25,000平方英尺以上土地的住宅用地比例<br>INDUS：每個城鎮非零售業務英畝的比例<br>CHAS：查爾斯河虛擬變量(如果束縛河，則為1；否則為0)<br>NOX：一氧化氮濃度(百萬分之一)<br>RM：每個住宅的平均房間數<br>AGE：1940年之前建造的自有住房的比例<br>DUS：到五個波士頓就業中心的加權距離<br>RAD：徑向公路的可達性指數<br>TAX：每10,000美元的全職財產稅率<br>PTRATIO:城鎮的師生比例<br>B：黑人比例<br>LSTAT：中下階級的比率<br>Target：城鎮房價的中間價格|
| 資料數  | 506筆 |
| 分析目標  | 建立一個已知特徵並預測波士頓未知房價的模型 |
| 分析分法  | 透過線性迴歸的方式來進行預測 |

## 1. 簡介

### 1.1 資料簡介

#### 1.1.1 資料來源

本次練習是採用sklearn所提供的波士頓房價資料

#### 1.1.2 特徵簡介

CRIM：人均城鎮犯罪率<br>ZN：劃為25,000平方英尺以上土地的住宅用地比例<br>INDUS：每個城鎮非零售業務英畝的比例<br>CHAS：查爾斯河虛擬變量(如果束縛河，則為1；否則為0)<br>NOX：一氧化氮濃度(百萬分之一)<br>RM：每個住宅的平均房間數<br>AGE：1940年之前建造的自有住房的比例<br>DUS：到五個波士頓就業中心的加權距離<br>RAD：徑向公路的可達性指數<br>TAX：每10,000美元的全職財產稅率<br>PTRATIO:城鎮的師生比例<br>B：黑人比例<br>LSTAT：中下階級的比率<br>Target：城鎮房價的中間價格

#### 1.1.3 資料的統計特徵
![20191214194214.png](https://upload.cc/i1/2023/05/16/rKRbNI.png)

### 1.2 分析工具和方法

#### 1.2.1 分析工具
線性迴歸是一種常用的統計分析方法，用於建立自變數和應變數之間的線性關係模型。透過迴歸分析，我們可以推斷自變數對應變數的影響程度，並用這種關係來預測未知的應變數數值。

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
![20191214204847.png](https://upload.cc/i1/2023/05/16/T5nFdt.png)

#### 2.2 資料分析圖表繪製

#### 2.2.1 熱力圖及相關矩陣
~~~python
    plt.figure(figsize=(15,10))
    plt.title('Correlation between features')
    sns.heatmap(df.corr(), annot=True, square=True)
    plt.show()
~~~
![20191214204847.png](https://upload.cc/i1/2023/05/16/O4ayYn.png)

#### 2.2.2 房價直方圖以及Q-Q plot
~~~python
    sns.distplot(df["Target"])
    plt.show()
    stats.probplot(df['Target'], plot=pylab)
    pylab.show()
~~~
![20191214204847.png](https://upload.cc/i1/2023/05/17/4WERrH.png)
![20191214204847.png](https://upload.cc/i1/2023/05/17/4WERrH.png)








