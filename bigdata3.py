import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

b_cancer = load_breast_cancer()

print(b_cancer.DESCR)
.. _breast_cancer_dataset:

Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        The mean, standard error, and "worst" or largest (mean of the three
        worst/largest values) of these features were computed for each image,
        resulting in 30 features.  For instance, field 0 is Mean Radius, field
        10 is Radius SE, field 20 is Worst Radius.

        - class:
                - WDBC-Malignant
                - WDBC-Benignb_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)

b_cancer_df['diagnosis'] = b_cancer.target

b_cancer_df.head()
print('유방암 진단 데이터셋 크기: ', b_cancer_df.shape)
유방암 진단 데이터셋 크기:  (569, 31)

b_cancer_df.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

b_cancer_scaled = scaler.fit_transform(b_cancer.data)

print(b_cancer.data[0])
 print(b_cancer_scaled[0])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#X, Y 설정하기
Y = b_cancer_df['diagnosis']
X = b_cancer_scaled

#훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

#로지스틱 회귀 분석: (1) 모델 생성
lr_b_cancer = LogisticRegression()

#로지스틱 회귀 분석: (2) 모델 훈련
lr_b_cancer.fit(X_train, Y_train)
LogisticRegression()

#로지스틱 회귀 분석: (3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr_b_cancer.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

#오차 행렬
confusion_matrix(Y_test, Y_predict)
array([[ 60,   3],
       [  1, 107]])
       
accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f}, F1: {3:.3f}'.format(accuracy,precision,recall,f1))
정확도: 0.977, 정밀도: 0.973, 재현율: 0.991, F1: 0.982

print('ROC_AUC: {0:.3f}'.format(roc_auc))
ROC_AUC: 0.972
import numpy as np
import pandas as pd

pd.__version__
1.3.5

#피처 이름 파일 읽어오기
feature_name_df = pd.read_csv('/features.txt', sep = '\s+', header = None, names = ['index', 'feature_name'], engine = 'python')

feature_name_df.head()
feature_name_df.shape
(561, 2)

#index 제거하고, feature_name만 리스트로 저장
feature_name = feature_name_df.iloc[:, 1].values.tolist()

feature_name[:5]
['tBodyAcc-mean()-X',
 'tBodyAcc-mean()-Y',
 'tBodyAcc-mean()-Z',
 'tBodyAcc-std()-X',
 'tBodyAcc-std()-Y']
 
 X_train = pd.read_csv('/X_train.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_train.columns = feature_name
X_test = pd.read_csv('/X_test.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_test.columns = feature_name
Y_train = pd.read_csv('/Y_train.txt', sep='\s+', header = None, names = ['action'], engine = 'python')
Y_test = pd.read_csv('/Y_test.txt', sep='\s+', header = None, names = ['action'], engine = 'python')

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
((7352, 561), (7352, 1), (2947, 561), (2947, 1))

X_train.head()
print(Y_train['action'].value_counts())
label_name_df = pd.read_csv('/activity_labels.txt', sep = '\s+', header = None, names = ['index', 'label'], engine = 'python')
label_name = label_name_df.iloc[:, 1].values.tolist()
