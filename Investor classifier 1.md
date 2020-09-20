```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


investor_data = pd.read_csv("/Users/school/Documents/investor_data-200412-235632.csv")
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
    </tr>
  </tbody>
</table>
</div>



In this kernel, we will be work with data to classify the type of investor of a syndicated revolver

Let's get accustomed to some terminologies:
Suppose you are working as the 'middleman' that raises money for a corporate client by gathering money from global financial institutions. The large corporate client also pays millions of dollar yearly(fee wallet) to global financial institutions for M&A fees, asset divestiture and advisory fees, debt and equity underwriting fees.

As a huge client for wall street firms, the corporation could leverage its status to demand cheap debt from the bank in the form of syndicated revolver (often they are below the bank's cost of capital, meaning the bank is losing money lending out the money)

However, the wall street banks must commit to these syndicated revolver in order to seek what they are actually after,  other more lucrative IBanking services. And traditionally, the amount of lucrative investment banking businesses awarded are proportional to their commitment to the revolver

Now we have some background of the industry, imagine your are working for a corporate client on raising these revolvers, can we predict which investors to invite, how much should we ask them to commit and anticipate their response?

Note that an accurate prediction to the investors' behaviour is important as if many banks decline to invest, your transaction will fall through, while if many do invests, you may find yourself in a situation where you have too many mouths to feed.

The data are past transactional data for thousands of data recording how banks reacted to the invitation to a new syndicated revolver.


```python
investor_data.hist(figsize = (10,7))
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_2_1.png)



```python
investor_data = investor_data[investor_data.total_fees >= 0]
investor_data.hist(figsize = (10,7))
plt.show()
```


![png](output_3_0.png)



```python
sns.countplot(y= 'investor', data=investor_data)
plt.show()
```


![png](output_4_0.png)



```python
sns.countplot(y= 'commit', data=investor_data)
plt.show()
```


![png](output_5_0.png)



```python
investor_data.groupby('investor').mean()
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
      <th>deal_size</th>
      <th>invite</th>
      <th>rating</th>
      <th>covenants</th>
      <th>total_fees</th>
      <th>fee_share</th>
    </tr>
    <tr>
      <th>investor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bank of America</th>
      <td>1023.648649</td>
      <td>154.824324</td>
      <td>3.985135</td>
      <td>1.371622</td>
      <td>106.456081</td>
      <td>16.225541</td>
    </tr>
    <tr>
      <th>Deutsche Bank</th>
      <td>1074.320652</td>
      <td>163.593750</td>
      <td>4.192255</td>
      <td>1.305027</td>
      <td>112.262228</td>
      <td>17.487024</td>
    </tr>
    <tr>
      <th>Goldman Sachs</th>
      <td>1038.028169</td>
      <td>154.985915</td>
      <td>4.026056</td>
      <td>1.364085</td>
      <td>108.548592</td>
      <td>16.348662</td>
    </tr>
    <tr>
      <th>MUFG Union</th>
      <td>1024.791086</td>
      <td>156.295265</td>
      <td>4.032730</td>
      <td>1.378830</td>
      <td>107.084958</td>
      <td>16.050557</td>
    </tr>
    <tr>
      <th>Wells Fargo</th>
      <td>1033.496162</td>
      <td>154.312631</td>
      <td>4.004885</td>
      <td>1.371249</td>
      <td>110.167481</td>
      <td>16.621075</td>
    </tr>
  </tbody>
</table>
</div>




```python
investor_data.commit.value_counts()
```




    Commit     5737
    Decline    1504
    Name: commit, dtype: int64




```python
investor_data.groupby('investor').commit.value_counts()
```




    investor         commit 
    Bank of America  Commit     1249
                     Decline     231
    Deutsche Bank    Commit      941
                     Decline     531
    Goldman Sachs    Commit     1064
                     Decline     356
    MUFG Union       Commit     1209
                     Decline     227
    Wells Fargo      Commit     1274
                     Decline     159
    Name: commit, dtype: int64




```python
investor_data.groupby('investor').commit.value_counts().plot(kind ='barh')
plt.show()
```


![png](output_9_0.png)



```python
investor_data.groupby('invite_tier').commit.value_counts().plot(kind ='barh')
plt.show()
```


![png](output_10_0.png)



```python
investor_data.groupby('commit').invite_tier.value_counts().plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa81892fc10>




![png](output_11_1.png)



```python
investor_data['tier_change']= np.where(investor_data.prior_tier == investor_data.invite_tier, 'None', 
                                       np.where(investor_data.prior_tier == 'Participant','Promoted','Demoted'))
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
    </tr>
  </tbody>
</table>
</div>




```python
investor_data.groupby('tier_change').commit.value_counts().plot(kind='barh')
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_13_1.png)



```python
investor_data[investor_data.tier_change =='None'].groupby('prior_tier').commit.value_counts()
```




    prior_tier   commit 
    Bookrunner   Commit     4329
                 Decline     462
    Participant  Commit      206
                 Decline      43
    Name: commit, dtype: int64




```python
investor_data[investor_data.investor =='Goldman Sachs'].groupby('commit').median()
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
      <th>deal_size</th>
      <th>invite</th>
      <th>rating</th>
      <th>covenants</th>
      <th>total_fees</th>
      <th>fee_share</th>
    </tr>
    <tr>
      <th>commit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Commit</th>
      <td>1100.0</td>
      <td>170.0</td>
      <td>3.0</td>
      <td>1.5</td>
      <td>107.0</td>
      <td>14.35</td>
    </tr>
    <tr>
      <th>Decline</th>
      <td>900.0</td>
      <td>100.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>68.0</td>
      <td>5.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
investor_data['fee_percent']= investor_data.fee_share / investor_data.total_fees
investor_data['invite_percent'] = investor_data.invite/ investor_data.deal_size
# these 2 are continuous so wont use count function, instead do following:
```


```python
# we will use the lmplot fucntion which creates a coloured scattered plot
# note the hue function define the categorical variable used for colour coding
sns.lmplot(x='deal_size',y='invite_percent', hue = 'commit', data = investor_data, fit_reg= False)
plt.show()
```


![png](output_17_0.png)



```python

```
