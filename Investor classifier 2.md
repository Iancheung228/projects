```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn

investor_data = pd.read_csv("/Users/school/Documents/investor_data_2-200408-143021.csv")
investor_data.head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>investor</th>
      <th>commit</th>
      <th>deal_size</th>
      <th>invite</th>
      <th>rating</th>
      <th>int_rate</th>
      <th>covenants</th>
      <th>total_fees</th>
      <th>fee_share</th>
      <th>prior_tier</th>
      <th>invite_tier</th>
      <th>tier_change</th>
      <th>fee_percent</th>
      <th>invite_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Goldman Sachs</td>
      <td>Commit</td>
      <td>300</td>
      <td>40</td>
      <td>2</td>
      <td>Market</td>
      <td>2</td>
      <td>30</td>
      <td>0.0</td>
      <td>Participant</td>
      <td>Bookrunner</td>
      <td>Promoted</td>
      <td>0.000000</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Deutsche Bank</td>
      <td>Decline</td>
      <td>1200</td>
      <td>140</td>
      <td>2</td>
      <td>Market</td>
      <td>2</td>
      <td>115</td>
      <td>20.1</td>
      <td>Bookrunner</td>
      <td>Participant</td>
      <td>Demoted</td>
      <td>0.174783</td>
      <td>0.116667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bank of America</td>
      <td>Commit</td>
      <td>900</td>
      <td>130</td>
      <td>3</td>
      <td>Market</td>
      <td>2</td>
      <td>98</td>
      <td>24.4</td>
      <td>Bookrunner</td>
      <td>Bookrunner</td>
      <td>None</td>
      <td>0.248980</td>
      <td>0.144444</td>
    </tr>
  </tbody>
</table>
</div>




```python
investor_data = investor_data.drop(['invite', 'fee_share', 'invite_tier'], axis=1)
investor_data.shape
```




    (7233, 11)




```python
investor_data = pd.get_dummies(investor_data)
investor_data.shape
```




    (7233, 21)




```python
investor_data = investor_data.drop('commit_Commit', axis =1)
investor_data.shape
```




    (7233, 20)




```python
target = investor_data.commit_Decline
inputs = investor_data.drop('commit_Decline', axis = 1)
```


```python
sns.countplot(y = 'commit_Decline' , data= investor_data)
plt.show()
```


![png](output_5_0.png)



```python
#stratified sampling
from sklearn.model_selection import train_test_split
split_list = train_test_split(inputs, target, test_size = 0.2, random_state=1, stratify=investor_data.commit_Decline)
```


```python
input_train, input_test, target_train, target_test = split_list
for item in [input_train, input_test, target_train, target_test]:
    print(item.shape)
```

    (5786, 19)
    (1447, 19)
    (5786,)
    (1447,)



```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

```


```python
pipelines = {
    'l1' : make_pipeline(StandardScaler(), LogisticRegression(penalty ='l1', random_state=1, solver='liblinear')),
    'l2' : make_pipeline(StandardScaler(), LogisticRegression(penalty ='l2', random_state=1, solver='liblinear')),
    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1))                
}
```


```python
for key , value in  pipelines.items():
    print(key, type(value))
```

    l1 <class 'sklearn.pipeline.Pipeline'>
    l2 <class 'sklearn.pipeline.Pipeline'>
    rf <class 'sklearn.pipeline.Pipeline'>
    gb <class 'sklearn.pipeline.Pipeline'>



```python
l1_hyperparameters ={
    'logisticregression__C' :[0.1, 1, 10]
}
l2_hyperparameters ={
    'logisticregression__C' :[0.1, 1, 10]
}
rf_hyperparameters ={
    'randomforestclassifier__n_estimators' :[100,200],
    'randomforestclassifier__max_features' :['auto', 0.3, 0.6]
}
gb_hyperparameters ={
    'gradientboostingclassifier__n_estimators' :[100,200], 
    'gradientboostingclassifier__learning_rate' :[0.05,0.1,0.2] ,
    'gradientboostingclassifier__max_depth' :[1,3,5] 
}
hyperparameters = {
    'l1' :l1_hyperparameters,
    'l2' :l2_hyperparameters,
    'rf' :rf_hyperparameters,
    'gb' :gb_hyperparameters
}
```


```python
for key in  ['l1', 'l2', 'rf', 'gb']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print(key, 'was found, and it is a grid')
        else:
            print(key, 'was found, but it is not a grid')
    else:
        print(key, 'was not found')
            
```

    l1 was found, and it is a grid
    l2 was found, and it is a grid
    rf was found, and it is a grid
    gb was found, and it is a grid



```python
from sklearn.model_selection import GridSearchCV
models = {}

for key in pipelines.keys():
    models[key] = GridSearchCV(pipelines[key], hyperparameters[key], cv = 5)
```


```python
for key in models:
    models[key].fit(input_train, target_train)
    print(key , 'is trained and tuned.')
```

    l1 is trained and tuned.
    l2 is trained and tuned.
    rf is trained and tuned.
    gb is trained and tuned.



```python
from sklearn.metrics import confusion_matrix
pred = models['l1'].predict(input_test)
print(confusion_matrix(target_test, pred))
```

    [[1124   22]
     [  23  278]]



```python
from sklearn.metrics import roc_curve, auc

#calculate ROC curve, unpack outputs, and print L1 AUROC
fpr, tpr, thresholds = roc_curve(target_test, pred)
print('l1')
print('AUROC =' , round(auc(fpr,tpr),3))
```

    l1
    AUROC = 0.952



```python
for key in models.keys() :
    pred =models[key].predict(input_test)
    fpr, tpr, thresholds = roc_curve(target_test, pred)
    print(key)
    print('AUROC =' , round(auc(fpr,tpr),4))
    print('---')

```

    l1
    AUROC = 0.9522
    ---
    l2
    AUROC = 0.9518
    ---
    rf
    AUROC = 0.9616
    ---
    gb
    AUROC = 0.9683
    ---



```python
# we see all 4 models are doing pretty well
```
