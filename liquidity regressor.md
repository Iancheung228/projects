```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

liquidity_data = pd.read_csv("/Users/school/Documents/liquidity_data-200412-220109.csv")
liquidity_data.head(3)

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
      <th>available_liquidity</th>
      <th>sp_score</th>
      <th>market_cap</th>
      <th>total_debt</th>
      <th>ltm_capex</th>
      <th>ltm_ebitda</th>
      <th>ltm_fcf</th>
      <th>ltm_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28694.04271</td>
      <td>2</td>
      <td>54856.1961</td>
      <td>84628.0</td>
      <td>-9262.0</td>
      <td>21387.00032</td>
      <td>9488.0</td>
      <td>170315.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24784.00051</td>
      <td>7</td>
      <td>209150.6401</td>
      <td>57909.0</td>
      <td>-2021.0</td>
      <td>15161.00019</td>
      <td>12105.0</td>
      <td>37727.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24142.00013</td>
      <td>6</td>
      <td>180108.3453</td>
      <td>32970.0</td>
      <td>-1817.0</td>
      <td>15818.99981</td>
      <td>12604.0</td>
      <td>192592.0</td>
    </tr>
  </tbody>
</table>
</div>



In this kernel, we use regression to see how much liquidity (corporate spending power) a company should maintain. Having too much liquidity means a high opportunity cost, while having too little results in trouble turning cash around. Note, if we use a linear model, it explains less than 0.5 of R^2. Trainning the model with machine learning is learning the "collective wisdom" of what companies are doing in the market. We will use the scikit learn library here.


```python
liquidity_data.describe()
# take a look at the below columns for an idea of what features we are working with
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
      <th>available_liquidity</th>
      <th>sp_score</th>
      <th>market_cap</th>
      <th>total_debt</th>
      <th>ltm_capex</th>
      <th>ltm_ebitda</th>
      <th>ltm_fcf</th>
      <th>ltm_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>802.000000</td>
      <td>802.000000</td>
      <td>802.000000</td>
      <td>802.000000</td>
      <td>802.00000</td>
      <td>802.000000</td>
      <td>802.000000</td>
      <td>802.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3884.952199</td>
      <td>3.017456</td>
      <td>41645.089870</td>
      <td>9040.720589</td>
      <td>-1200.91799</td>
      <td>3455.752891</td>
      <td>1772.335973</td>
      <td>20420.383638</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4267.893247</td>
      <td>1.851461</td>
      <td>74046.522440</td>
      <td>12112.161513</td>
      <td>2066.29159</td>
      <td>5679.466199</td>
      <td>4207.345101</td>
      <td>39483.422972</td>
    </tr>
    <tr>
      <th>min</th>
      <td>267.000000</td>
      <td>0.000000</td>
      <td>4282.810112</td>
      <td>0.000000</td>
      <td>-15858.00000</td>
      <td>-6530.000000</td>
      <td>-4888.000000</td>
      <td>503.586000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1288.328992</td>
      <td>2.000000</td>
      <td>10082.091010</td>
      <td>2125.297000</td>
      <td>-1241.00000</td>
      <td>936.200000</td>
      <td>232.923000</td>
      <td>3791.525000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2395.700000</td>
      <td>3.000000</td>
      <td>19349.403650</td>
      <td>4562.000000</td>
      <td>-512.00000</td>
      <td>1695.700032</td>
      <td>682.110000</td>
      <td>8587.166000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4512.458596</td>
      <td>4.000000</td>
      <td>41154.826240</td>
      <td>10478.000000</td>
      <td>-190.00000</td>
      <td>3706.000000</td>
      <td>1788.000000</td>
      <td>18816.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28694.042710</td>
      <td>10.000000</td>
      <td>777070.706700</td>
      <td>87032.000000</td>
      <td>-5.04900</td>
      <td>69715.000320</td>
      <td>53244.000000</td>
      <td>487511.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
```


```python
target = liquidity_data.available_liquidity

inputs = liquidity_data.drop('available_liquidity', axis =1)

```


```python
target.head(1)

```




    0    28694.04271
    Name: available_liquidity, dtype: float64




```python
inputs.head(1)
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
      <th>sp_score</th>
      <th>market_cap</th>
      <th>total_debt</th>
      <th>ltm_capex</th>
      <th>ltm_ebitda</th>
      <th>ltm_fcf</th>
      <th>ltm_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>54856.1961</td>
      <td>84628.0</td>
      <td>-9262.0</td>
      <td>21387.00032</td>
      <td>9488.0</td>
      <td>170315.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = train_test_split(inputs, target, test_size=0.2, random_state=1)

```


```python
print(type(results))
print(len(results))
print('---')
for item in results:
    print(item.shape)
    
    # the first 2 object are dataframes, last two are series
```

    <class 'list'>
    4
    ---
    (641, 7)
    (161, 7)
    (641,)
    (161,)



```python
#Define List 
example_list = [1,2,3]

#unpack list
one, two , three = example_list

print(one)
print(two)
print(three)
```

    1
    2
    3



```python
input_train, input_test ,target_train, target_test = results

print(input_train.shape)
print(input_test.shape)
print(target_train.shape)
print(target_test.shape)
```

    (641, 7)
    (161, 7)
    (641,)
    (161,)



```python
input_train.head(1)
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
      <th>sp_score</th>
      <th>market_cap</th>
      <th>total_debt</th>
      <th>ltm_capex</th>
      <th>ltm_ebitda</th>
      <th>ltm_fcf</th>
      <th>ltm_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>309</th>
      <td>3</td>
      <td>16764.94643</td>
      <td>1887.019</td>
      <td>-483.002</td>
      <td>1905.83296</td>
      <td>859.71</td>
      <td>8653.205</td>
    </tr>
  </tbody>
</table>
</div>




```python
input_test.head(1)
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
      <th>sp_score</th>
      <th>market_cap</th>
      <th>total_debt</th>
      <th>ltm_capex</th>
      <th>ltm_ebitda</th>
      <th>ltm_fcf</th>
      <th>ltm_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>172479.4798</td>
      <td>24842.0</td>
      <td>-1674.0</td>
      <td>10841.00006</td>
      <td>6830.0</td>
      <td>39929.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
target_train.head(1)

```




    309    1227.539
    Name: available_liquidity, dtype: float64




```python
target_test.head(1)
```




    8    17708.00026
    Name: available_liquidity, dtype: float64




```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipelines ={
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=1)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=1))
    
}
# recall that lasso is L1, Ridge is L2, and elastic net is a blend between the 2!
```


```python
# dictionary_name['key name'] =value

from sklearn.linear_model import ElasticNet

pipelines['enet']=make_pipeline(StandardScaler(),ElasticNet(random_state=1))
```


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

pipelines['rf']=make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1))
pipelines['gb']=make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1))

#recall that random forest is bagging, while GB is boosting within the ensemble method.
```


```python
for key, value in pipelines.items():
    print(key,type(value))
```

    lasso <class 'sklearn.pipeline.Pipeline'>
    ridge <class 'sklearn.pipeline.Pipeline'>
    enet <class 'sklearn.pipeline.Pipeline'>
    rf <class 'sklearn.pipeline.Pipeline'>
    gb <class 'sklearn.pipeline.Pipeline'>



```python
lasso_hyperparameters ={
    'lasso__alpha': [0.01, 0.05,  0.1, 0.5, 1 , 5]
}
ridge_hyperparameters = {
    'ridge__alpha' : [0.01, 0.05,  0.1, 0.5, 1 , 5]
}
enet_hyperparameters = {
    'elasticnet__alpha' : [0.01, 0.05,  0.1, 0.5, 1 , 5],
    'elasticnet__l1_ratio' : [0.1,0.3,0.5,0.7,0.9]
}
```


```python
rf_hyperparameters ={
    'randomforestregressor__n_estimators' :[100,200],
    'randomforestregressor__max_features' :['auto', 0.3, 0.6]
}

gb_hyperparameters ={
    'gradientboostingregressor__n_estimators' :[100,200],
    'gradientboostingregressor__learning_rate' :[0.05,0.1,0.2],
    'gradientboostingregressor__max_depth' :[1,3,5]
}
```


```python
hyperparameter_grids ={
    'lasso': lasso_hyperparameters,
    'ridge': ridge_hyperparameters,
    'enet': enet_hyperparameters,
    'rf': rf_hyperparameters,
    'gb': gb_hyperparameters
}
```


```python
for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameter_grids:
        if type (hyperparameter_grids[key]) is dict:
            print(key, 'was found, and it is a grid')
        else:
            print(key, 'was found, and it is not a grid')
    else:
        print(key, 'was not found')
```

    enet was found, and it is a grid
    gb was found, and it is a grid
    ridge was found, and it is a grid
    rf was found, and it is a grid
    lasso was found, and it is a grid



```python
# want to create untrainned model for each model class
from sklearn.model_selection import GridSearchCV

untrained_lasso_model = GridSearchCV(pipelines['lasso'], hyperparameter_grids['lasso'],cv=5)
```


```python
print(pipelines.keys())
print("---")
print(hyperparameter_grids.keys())
```

    dict_keys(['lasso', 'ridge', 'enet', 'rf', 'gb'])
    ---
    dict_keys(['lasso', 'ridge', 'enet', 'rf', 'gb'])



```python
models ={
    
}
for key in pipelines.keys():
     models[key] = GridSearchCV(pipelines[key], hyperparameter_grids[key], cv=5)
        
models.keys()
```




    dict_keys(['lasso', 'ridge', 'enet', 'rf', 'gb'])




```python
#second step, trainning data and find optimal config
models['lasso'].fit(input_train, target_train)
```




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('standardscaler', StandardScaler()),
                                           ('lasso', Lasso(random_state=1))]),
                 param_grid={'lasso__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5]})




```python
for key in models.keys():
    models[key] = models[key].fit(input_train, target_train)
    print(key, 'is trainned and tuned')
```

    lasso is trainned and tuned
    ridge is trainned and tuned
    enet is trainned and tuned
    rf is trainned and tuned
    gb is trainned and tuned



```python
from sklearn.metrics import r2_score, mean_absolute_error

lasso_preds = models['lasso'].predict(input_test)
print('R-squared:', round (r2_score(target_test, lasso_preds), 3))
print('MAE:', round (mean_absolute_error(target_test, lasso_preds), 3))
```

    R-squared: 0.498
    MAE: 1710.083



```python
for key in models:
    preds =models[key].predict(input_test)
    # input_test is the input without the liquidity
    print(key)
    print('R-squared:', round (r2_score(target_test,preds), 3))
    #target_test is the liquidity of the input_test, aka the correct, empirical answer
    print('MAE:', round (mean_absolute_error(target_test, preds), 3))
    print('---')
#the preds is a temporary variable, doesnt exist outside for loop

## we see that gb is the best performing model with a R-squared of 0.855
```

    lasso
    R-squared: 0.498
    MAE: 1710.083
    ---
    ridge
    R-squared: 0.5
    MAE: 1708.062
    ---
    enet
    R-squared: 0.501
    MAE: 1706.04
    ---
    rf
    R-squared: 0.85
    MAE: 929.691
    ---
    gb
    R-squared: 0.855
    MAE: 588.249
    ---



```python
#Make predictions with test data 
preds = models['lasso'].predict(input_test)
#Plot the predictions on the x axis and actuals on y axis
plt.scatter(preds, target_test)

#label axis and show graph
plt.xlabel('predicted')
plt.ylabel("actual")
plt.show()
```


![png](output_30_0.png)



```python
#Make predictions with test data 
preds = models['gb'].predict(input_test)
#Plot the predictions on the x axis and actuals on y axis
plt.scatter(preds, target_test)

#label axis and show graph
plt.xlabel('predicted')
plt.ylabel("actual")
plt.show()
```


![png](output_31_0.png)



```python

```
