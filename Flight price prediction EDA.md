```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


```


```python
train_data = pd.read_excel("/Users/school/Documents/Flight_Data_Train.xlsx")
```


```python
pd.set_option('display.max_columns', None)
```


```python
train_data.head()
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
      <th>Airline</th>
      <th>Date_of_Journey</th>
      <th>Source</th>
      <th>Destination</th>
      <th>Route</th>
      <th>Dep_Time</th>
      <th>Arrival_Time</th>
      <th>Duration</th>
      <th>Total_Stops</th>
      <th>Additional_Info</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IndiGo</td>
      <td>24/03/2019</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>BLR → DEL</td>
      <td>22:20</td>
      <td>01:10 22 Mar</td>
      <td>2h 50m</td>
      <td>non-stop</td>
      <td>No info</td>
      <td>3897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Air India</td>
      <td>1/05/2019</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → IXR → BBI → BLR</td>
      <td>05:50</td>
      <td>13:15</td>
      <td>7h 25m</td>
      <td>2 stops</td>
      <td>No info</td>
      <td>7662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jet Airways</td>
      <td>9/06/2019</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>DEL → LKO → BOM → COK</td>
      <td>09:25</td>
      <td>04:25 10 Jun</td>
      <td>19h</td>
      <td>2 stops</td>
      <td>No info</td>
      <td>13882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IndiGo</td>
      <td>12/05/2019</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → NAG → BLR</td>
      <td>18:05</td>
      <td>23:30</td>
      <td>5h 25m</td>
      <td>1 stop</td>
      <td>No info</td>
      <td>6218</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IndiGo</td>
      <td>01/03/2019</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>BLR → NAG → DEL</td>
      <td>16:50</td>
      <td>21:35</td>
      <td>4h 45m</td>
      <td>1 stop</td>
      <td>No info</td>
      <td>13302</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.info

```




    <bound method DataFrame.info of            Airline Date_of_Journey    Source Destination  \
    0           IndiGo      24/03/2019  Banglore   New Delhi   
    1        Air India       1/05/2019   Kolkata    Banglore   
    2      Jet Airways       9/06/2019     Delhi      Cochin   
    3           IndiGo      12/05/2019   Kolkata    Banglore   
    4           IndiGo      01/03/2019  Banglore   New Delhi   
    ...            ...             ...       ...         ...   
    10678     Air Asia       9/04/2019   Kolkata    Banglore   
    10679    Air India      27/04/2019   Kolkata    Banglore   
    10680  Jet Airways      27/04/2019  Banglore       Delhi   
    10681      Vistara      01/03/2019  Banglore   New Delhi   
    10682    Air India       9/05/2019     Delhi      Cochin   
    
                           Route Dep_Time  Arrival_Time Duration Total_Stops  \
    0                  BLR → DEL    22:20  01:10 22 Mar   2h 50m    non-stop   
    1      CCU → IXR → BBI → BLR    05:50         13:15   7h 25m     2 stops   
    2      DEL → LKO → BOM → COK    09:25  04:25 10 Jun      19h     2 stops   
    3            CCU → NAG → BLR    18:05         23:30   5h 25m      1 stop   
    4            BLR → NAG → DEL    16:50         21:35   4h 45m      1 stop   
    ...                      ...      ...           ...      ...         ...   
    10678              CCU → BLR    19:55         22:25   2h 30m    non-stop   
    10679              CCU → BLR    20:45         23:20   2h 35m    non-stop   
    10680              BLR → DEL    08:20         11:20       3h    non-stop   
    10681              BLR → DEL    11:30         14:10   2h 40m    non-stop   
    10682  DEL → GOI → BOM → COK    10:55         19:15   8h 20m     2 stops   
    
          Additional_Info  Price  
    0             No info   3897  
    1             No info   7662  
    2             No info  13882  
    3             No info   6218  
    4             No info  13302  
    ...               ...    ...  
    10678         No info   4107  
    10679         No info   4145  
    10680         No info   7229  
    10681         No info  12648  
    10682         No info  11753  
    
    [10683 rows x 11 columns]>




```python
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10683 entries, 0 to 10682
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Airline          10683 non-null  object
     1   Date_of_Journey  10683 non-null  object
     2   Source           10683 non-null  object
     3   Destination      10683 non-null  object
     4   Route            10682 non-null  object
     5   Dep_Time         10683 non-null  object
     6   Arrival_Time     10683 non-null  object
     7   Duration         10683 non-null  object
     8   Total_Stops      10682 non-null  object
     9   Additional_Info  10683 non-null  object
     10  Price            10683 non-null  int64 
    dtypes: int64(1), object(10)
    memory usage: 918.2+ KB



```python
train_data.shape
```




    (10683, 11)




```python
train_data["Duration"].value_counts()
```




    2h 50m     550
    1h 30m     386
    2h 55m     337
    2h 45m     337
    2h 35m     329
              ... 
    35h 20m      1
    3h 25m       1
    40h 20m      1
    36h 25m      1
    41h 20m      1
    Name: Duration, Length: 368, dtype: int64




```python
#dropping the NaN values
train_data.dropna(inplace = True)
```


```python
train_data.shape
```




    (10682, 11)




```python
train_data.isnull().sum()

```




    Airline            0
    Date_of_Journey    0
    Source             0
    Destination        0
    Route              0
    Dep_Time           0
    Arrival_Time       0
    Duration           0
    Total_Stops        0
    Additional_Info    0
    Price              0
    dtype: int64




```python
####EDA 

```


```python
#Here we will be creating a new column call "Journey_day" by using the following function 
train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format ="%d/%m/%Y").dt.day
```


```python
# Similarly, we do the same for the month, notice the 2 diff ways of accessing the DF
train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
```


```python
train_data.head()
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
      <th>Airline</th>
      <th>Date_of_Journey</th>
      <th>Source</th>
      <th>Destination</th>
      <th>Route</th>
      <th>Dep_Time</th>
      <th>Arrival_Time</th>
      <th>Duration</th>
      <th>Total_Stops</th>
      <th>Additional_Info</th>
      <th>Price</th>
      <th>Journey_day</th>
      <th>Journey_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IndiGo</td>
      <td>24/03/2019</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>BLR → DEL</td>
      <td>22:20</td>
      <td>01:10 22 Mar</td>
      <td>2h 50m</td>
      <td>non-stop</td>
      <td>No info</td>
      <td>3897</td>
      <td>24</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Air India</td>
      <td>1/05/2019</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → IXR → BBI → BLR</td>
      <td>05:50</td>
      <td>13:15</td>
      <td>7h 25m</td>
      <td>2 stops</td>
      <td>No info</td>
      <td>7662</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jet Airways</td>
      <td>9/06/2019</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>DEL → LKO → BOM → COK</td>
      <td>09:25</td>
      <td>04:25 10 Jun</td>
      <td>19h</td>
      <td>2 stops</td>
      <td>No info</td>
      <td>13882</td>
      <td>9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IndiGo</td>
      <td>12/05/2019</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → NAG → BLR</td>
      <td>18:05</td>
      <td>23:30</td>
      <td>5h 25m</td>
      <td>1 stop</td>
      <td>No info</td>
      <td>6218</td>
      <td>12</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IndiGo</td>
      <td>01/03/2019</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>BLR → NAG → DEL</td>
      <td>16:50</td>
      <td>21:35</td>
      <td>4h 45m</td>
      <td>1 stop</td>
      <td>No info</td>
      <td>13302</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# now the "Date of journey" is uesless
train_data.drop("Date_of_Journey", axis =1, inplace=True)
```


```python
# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time

# Extracting Hours
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

# Extracting Minutes
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute

# Now we can drop Dep_Time as it is of no use
train_data.drop(["Dep_Time"], axis = 1, inplace = True)
```


```python
# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time

# Extracting Hours
train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour

# Extracting Minutes
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
train_data.drop(["Arrival_Time"], axis = 1, inplace = True)


```


```python

# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(train_data["Duration"])
#print(train_data["Duration"])
#print(duration)

for item in range(len(duration)):
    if len(duration[item].split()) !=2:
        if "h" in duration[item]:
            duration[item] = duration[item] + ' 0m'
        else:
            duration[item] = '0h ' + duration[item]
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
```


```python
'2h 50m'.split()
len('2h 50m'.split())
```




    2




```python
# Adding duration_hours and duration_mins list to train_data dataframe

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins
```


```python
train_data.drop(["Duration"], axis = 1, inplace = True)
```


```python
train_data.head()
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
      <th>Airline</th>
      <th>Source</th>
      <th>Destination</th>
      <th>Route</th>
      <th>Total_Stops</th>
      <th>Additional_Info</th>
      <th>Price</th>
      <th>Journey_day</th>
      <th>Journey_month</th>
      <th>Dep_hour</th>
      <th>Dep_min</th>
      <th>Arrival_hour</th>
      <th>Arrival_min</th>
      <th>Duration_hours</th>
      <th>Duration_mins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IndiGo</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>BLR → DEL</td>
      <td>non-stop</td>
      <td>No info</td>
      <td>3897</td>
      <td>24</td>
      <td>3</td>
      <td>22</td>
      <td>20</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Air India</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → IXR → BBI → BLR</td>
      <td>2 stops</td>
      <td>No info</td>
      <td>7662</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>50</td>
      <td>13</td>
      <td>15</td>
      <td>7</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jet Airways</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>DEL → LKO → BOM → COK</td>
      <td>2 stops</td>
      <td>No info</td>
      <td>13882</td>
      <td>9</td>
      <td>6</td>
      <td>9</td>
      <td>25</td>
      <td>4</td>
      <td>25</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IndiGo</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → NAG → BLR</td>
      <td>1 stop</td>
      <td>No info</td>
      <td>6218</td>
      <td>12</td>
      <td>5</td>
      <td>18</td>
      <td>5</td>
      <td>23</td>
      <td>30</td>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IndiGo</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>BLR → NAG → DEL</td>
      <td>1 stop</td>
      <td>No info</td>
      <td>13302</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>50</td>
      <td>21</td>
      <td>35</td>
      <td>4</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data["Airline"].value_counts()
```




    Jet Airways                          3849
    IndiGo                               2053
    Air India                            1751
    Multiple carriers                    1196
    SpiceJet                              818
    Vistara                               479
    Air Asia                              319
    GoAir                                 194
    Multiple carriers Premium economy      13
    Jet Airways Business                    6
    Vistara Premium economy                 3
    Trujet                                  1
    Name: Airline, dtype: int64




```python
sns.catplot( y = "Price", x = "Airline", data= train_data.sort_values("Price", ascending =False), kind = "boxen", height =6, aspect=3)
plt.show()

```


![png](output_24_0.png)



```python
# we clearly see that jet Airways Business most likely offer more premium flights than its counterparts. It is 
# sensible to say that the Airlines are of Nomiinal category (no order) and hence we perform the OneHotEncoding.

Airline = train_data[['Airline']]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()
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
      <th>Airline_Air India</th>
      <th>Airline_GoAir</th>
      <th>Airline_IndiGo</th>
      <th>Airline_Jet Airways</th>
      <th>Airline_Jet Airways Business</th>
      <th>Airline_Multiple carriers</th>
      <th>Airline_Multiple carriers Premium economy</th>
      <th>Airline_SpiceJet</th>
      <th>Airline_Trujet</th>
      <th>Airline_Vistara</th>
      <th>Airline_Vistara Premium economy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.Source.value_counts()
```




    Delhi       4536
    Kolkata     2871
    Banglore    2197
    Mumbai       697
    Chennai      381
    Name: Source, dtype: int64




```python
# Source vs Price

sns.catplot(y = "Price", x = "Source", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()
```


![png](output_27_0.png)



```python
# As Source is Nominal Categorical data we will perform OneHotEncoding

Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()
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
      <th>Source_Chennai</th>
      <th>Source_Delhi</th>
      <th>Source_Kolkata</th>
      <th>Source_Mumbai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

train_data["Destination"].value_counts()
```




    Cochin       4536
    Banglore     2871
    Delhi        1265
    New Delhi     932
    Hyderabad     697
    Kolkata       381
    Name: Destination, dtype: int64




```python

# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()
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
      <th>Destination_Cochin</th>
      <th>Destination_Delhi</th>
      <th>Destination_Hyderabad</th>
      <th>Destination_Kolkata</th>
      <th>Destination_New Delhi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data["Route"]
```




    0                    BLR → DEL
    1        CCU → IXR → BBI → BLR
    2        DEL → LKO → BOM → COK
    3              CCU → NAG → BLR
    4              BLR → NAG → DEL
                     ...          
    10678                CCU → BLR
    10679                CCU → BLR
    10680                BLR → DEL
    10681                BLR → DEL
    10682    DEL → GOI → BOM → COK
    Name: Route, Length: 10682, dtype: object




```python

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other

train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
```


```python
train_data["Total_Stops"].value_counts()
```




    1 stop      5625
    non-stop    3491
    2 stops     1520
    3 stops       45
    4 stops        1
    Name: Total_Stops, dtype: int64




```python
# Total Stops is actually ordinal categorical type so we need to perform LabelEncoder
# Here Values are assigned with corresponding keys
train_data.replace({"non-stop" : 0, "1 stop" : 1, "2 stops" : 2, "3 stops" :3, "4 stops" :4} , inplace = True)
```


```python
# Concatenate dataframe --> train_data + Airline + Source + Destination

data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)
```


```python
data_train.shape
```




    (10682, 33)




```python
data_train.head()
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
      <th>Airline</th>
      <th>Source</th>
      <th>Destination</th>
      <th>Total_Stops</th>
      <th>Price</th>
      <th>Journey_day</th>
      <th>Journey_month</th>
      <th>Dep_hour</th>
      <th>Dep_min</th>
      <th>Arrival_hour</th>
      <th>Arrival_min</th>
      <th>Duration_hours</th>
      <th>Duration_mins</th>
      <th>Airline_Air India</th>
      <th>Airline_GoAir</th>
      <th>Airline_IndiGo</th>
      <th>Airline_Jet Airways</th>
      <th>Airline_Jet Airways Business</th>
      <th>Airline_Multiple carriers</th>
      <th>Airline_Multiple carriers Premium economy</th>
      <th>Airline_SpiceJet</th>
      <th>Airline_Trujet</th>
      <th>Airline_Vistara</th>
      <th>Airline_Vistara Premium economy</th>
      <th>Source_Chennai</th>
      <th>Source_Delhi</th>
      <th>Source_Kolkata</th>
      <th>Source_Mumbai</th>
      <th>Destination_Cochin</th>
      <th>Destination_Delhi</th>
      <th>Destination_Hyderabad</th>
      <th>Destination_Kolkata</th>
      <th>Destination_New Delhi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IndiGo</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>0</td>
      <td>3897</td>
      <td>24</td>
      <td>3</td>
      <td>22</td>
      <td>20</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Air India</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>2</td>
      <td>7662</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>50</td>
      <td>13</td>
      <td>15</td>
      <td>7</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jet Airways</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>2</td>
      <td>13882</td>
      <td>9</td>
      <td>6</td>
      <td>9</td>
      <td>25</td>
      <td>4</td>
      <td>25</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IndiGo</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>1</td>
      <td>6218</td>
      <td>12</td>
      <td>5</td>
      <td>18</td>
      <td>5</td>
      <td>23</td>
      <td>30</td>
      <td>5</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IndiGo</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>1</td>
      <td>13302</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>50</td>
      <td>21</td>
      <td>35</td>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
```


```python
data_train.head()
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
      <th>Total_Stops</th>
      <th>Price</th>
      <th>Journey_day</th>
      <th>Journey_month</th>
      <th>Dep_hour</th>
      <th>Dep_min</th>
      <th>Arrival_hour</th>
      <th>Arrival_min</th>
      <th>Duration_hours</th>
      <th>Duration_mins</th>
      <th>Airline_Air India</th>
      <th>Airline_GoAir</th>
      <th>Airline_IndiGo</th>
      <th>Airline_Jet Airways</th>
      <th>Airline_Jet Airways Business</th>
      <th>Airline_Multiple carriers</th>
      <th>Airline_Multiple carriers Premium economy</th>
      <th>Airline_SpiceJet</th>
      <th>Airline_Trujet</th>
      <th>Airline_Vistara</th>
      <th>Airline_Vistara Premium economy</th>
      <th>Source_Chennai</th>
      <th>Source_Delhi</th>
      <th>Source_Kolkata</th>
      <th>Source_Mumbai</th>
      <th>Destination_Cochin</th>
      <th>Destination_Delhi</th>
      <th>Destination_Hyderabad</th>
      <th>Destination_Kolkata</th>
      <th>Destination_New Delhi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3897</td>
      <td>24</td>
      <td>3</td>
      <td>22</td>
      <td>20</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7662</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>50</td>
      <td>13</td>
      <td>15</td>
      <td>7</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>13882</td>
      <td>9</td>
      <td>6</td>
      <td>9</td>
      <td>25</td>
      <td>4</td>
      <td>25</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>6218</td>
      <td>12</td>
      <td>5</td>
      <td>18</td>
      <td>5</td>
      <td>23</td>
      <td>30</td>
      <td>5</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13302</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>50</td>
      <td>21</td>
      <td>35</td>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_train.shape
```




    (10682, 30)




```python
#### TEST
test_data = pd.read_excel(r"/Users/school/Documents/Flight_Test_set.xlsx")
```


```python
test_data.head()
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
      <th>Airline</th>
      <th>Date_of_Journey</th>
      <th>Source</th>
      <th>Destination</th>
      <th>Route</th>
      <th>Dep_Time</th>
      <th>Arrival_Time</th>
      <th>Duration</th>
      <th>Total_Stops</th>
      <th>Additional_Info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jet Airways</td>
      <td>6/06/2019</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>DEL → BOM → COK</td>
      <td>17:30</td>
      <td>04:25 07 Jun</td>
      <td>10h 55m</td>
      <td>1 stop</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiGo</td>
      <td>12/05/2019</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>CCU → MAA → BLR</td>
      <td>06:20</td>
      <td>10:20</td>
      <td>4h</td>
      <td>1 stop</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jet Airways</td>
      <td>21/05/2019</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>DEL → BOM → COK</td>
      <td>19:15</td>
      <td>19:00 22 May</td>
      <td>23h 45m</td>
      <td>1 stop</td>
      <td>In-flight meal not included</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Multiple carriers</td>
      <td>21/05/2019</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>DEL → BOM → COK</td>
      <td>08:00</td>
      <td>21:00</td>
      <td>13h</td>
      <td>1 stop</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Air Asia</td>
      <td>24/06/2019</td>
      <td>Banglore</td>
      <td>Delhi</td>
      <td>BLR → DEL</td>
      <td>23:55</td>
      <td>02:45 25 Jun</td>
      <td>2h 50m</td>
      <td>non-stop</td>
      <td>No info</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()


print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)
```

    Test data Info
    ---------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2671 entries, 0 to 2670
    Data columns (total 10 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Airline          2671 non-null   object
     1   Date_of_Journey  2671 non-null   object
     2   Source           2671 non-null   object
     3   Destination      2671 non-null   object
     4   Route            2671 non-null   object
     5   Dep_Time         2671 non-null   object
     6   Arrival_Time     2671 non-null   object
     7   Duration         2671 non-null   object
     8   Total_Stops      2671 non-null   object
     9   Additional_Info  2671 non-null   object
    dtypes: object(10)
    memory usage: 208.8+ KB
    None
    
    
    Null values :
    ---------------------------------------------------------------------------
    Airline            0
    Date_of_Journey    0
    Source             0
    Destination        0
    Route              0
    Dep_Time           0
    Arrival_Time       0
    Duration           0
    Total_Stops        0
    Additional_Info    0
    dtype: int64
    Airline
    ---------------------------------------------------------------------------
    Jet Airways                          897
    IndiGo                               511
    Air India                            440
    Multiple carriers                    347
    SpiceJet                             208
    Vistara                              129
    Air Asia                              86
    GoAir                                 46
    Multiple carriers Premium economy      3
    Vistara Premium economy                2
    Jet Airways Business                   2
    Name: Airline, dtype: int64
    
    Source
    ---------------------------------------------------------------------------
    Delhi       1145
    Kolkata      710
    Banglore     555
    Mumbai       186
    Chennai       75
    Name: Source, dtype: int64
    
    Destination
    ---------------------------------------------------------------------------
    Cochin       1145
    Banglore      710
    Delhi         317
    New Delhi     238
    Hyderabad     186
    Kolkata        75
    Name: Destination, dtype: int64
    
    
    Shape of test data :  (2671, 28)



```python
data_test.head()
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
      <th>Total_Stops</th>
      <th>Journey_day</th>
      <th>Journey_month</th>
      <th>Dep_hour</th>
      <th>Dep_min</th>
      <th>Arrival_hour</th>
      <th>Arrival_min</th>
      <th>Duration_hours</th>
      <th>Duration_mins</th>
      <th>Air India</th>
      <th>GoAir</th>
      <th>IndiGo</th>
      <th>Jet Airways</th>
      <th>Jet Airways Business</th>
      <th>Multiple carriers</th>
      <th>Multiple carriers Premium economy</th>
      <th>SpiceJet</th>
      <th>Vistara</th>
      <th>Vistara Premium economy</th>
      <th>Chennai</th>
      <th>Delhi</th>
      <th>Kolkata</th>
      <th>Mumbai</th>
      <th>Cochin</th>
      <th>Delhi</th>
      <th>Hyderabad</th>
      <th>Kolkata</th>
      <th>New Delhi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>17</td>
      <td>30</td>
      <td>4</td>
      <td>25</td>
      <td>10</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>12</td>
      <td>5</td>
      <td>6</td>
      <td>20</td>
      <td>10</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>21</td>
      <td>5</td>
      <td>19</td>
      <td>15</td>
      <td>19</td>
      <td>0</td>
      <td>23</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>21</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>24</td>
      <td>6</td>
      <td>23</td>
      <td>55</td>
      <td>2</td>
      <td>45</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_train.shape
```




    (10682, 30)




```python

data_train.columns
```




    Index(['Total_Stops', 'Price', 'Journey_day', 'Journey_month', 'Dep_hour',
           'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
           'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
           'Airline_Jet Airways', 'Airline_Jet Airways Business',
           'Airline_Multiple carriers',
           'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
           'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
           'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
           'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
           'Destination_Kolkata', 'Destination_New Delhi'],
          dtype='object')




```python
X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()
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
      <th>Total_Stops</th>
      <th>Journey_day</th>
      <th>Journey_month</th>
      <th>Dep_hour</th>
      <th>Dep_min</th>
      <th>Arrival_hour</th>
      <th>Arrival_min</th>
      <th>Duration_hours</th>
      <th>Duration_mins</th>
      <th>Airline_Air India</th>
      <th>Airline_GoAir</th>
      <th>Airline_IndiGo</th>
      <th>Airline_Jet Airways</th>
      <th>Airline_Jet Airways Business</th>
      <th>Airline_Multiple carriers</th>
      <th>Airline_Multiple carriers Premium economy</th>
      <th>Airline_SpiceJet</th>
      <th>Airline_Trujet</th>
      <th>Airline_Vistara</th>
      <th>Airline_Vistara Premium economy</th>
      <th>Source_Chennai</th>
      <th>Source_Delhi</th>
      <th>Source_Kolkata</th>
      <th>Source_Mumbai</th>
      <th>Destination_Cochin</th>
      <th>Destination_Delhi</th>
      <th>Destination_Hyderabad</th>
      <th>Destination_Kolkata</th>
      <th>Destination_New Delhi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>24</td>
      <td>3</td>
      <td>22</td>
      <td>20</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>50</td>
      <td>13</td>
      <td>15</td>
      <td>7</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9</td>
      <td>6</td>
      <td>9</td>
      <td>25</td>
      <td>4</td>
      <td>25</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>12</td>
      <td>5</td>
      <td>18</td>
      <td>5</td>
      <td>23</td>
      <td>30</td>
      <td>5</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>50</td>
      <td>21</td>
      <td>35</td>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = data_train.iloc[:, 1]
y.head()
```




    0     3897
    1     7662
    2    13882
    3     6218
    4    13302
    Name: Price, dtype: int64




```python
# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")

plt.show()
```


![png](output_49_0.png)



```python
# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)
```




    ExtraTreesRegressor()




```python
print(selection.feature_importances_)
```

    [2.33790588e-01 1.42495217e-01 5.41494706e-02 2.44526843e-02
     2.10055373e-02 2.79215036e-02 1.89948417e-02 1.21679891e-01
     1.78661382e-02 9.26359614e-03 2.09135341e-03 1.78758861e-02
     1.40824580e-01 6.69412875e-02 1.81995034e-02 8.50506849e-04
     3.19959416e-03 1.12800146e-04 5.00495160e-03 8.68093172e-05
     4.55504104e-04 9.78239635e-03 3.24179889e-03 5.82870985e-03
     9.75220979e-03 1.26326867e-02 6.12897648e-03 5.34128232e-04
     2.48368491e-02]



```python
#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
```


![png](output_52_0.png)



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```


```python
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

```




    RandomForestRegressor()




```python

y_pred = reg_rf.predict(X_test)
```


```python
reg_rf.score(X_train, y_train)
```




    0.9530543293400993




```python
reg_rf.score(X_test, y_test)
```




    0.7979176783247053




```python
sns.distplot(y_test-y_pred)
plt.show()
```


![png](output_58_0.png)



```python
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
```


![png](output_59_0.png)



```python
from sklearn import metrics
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

    MAE: 1175.6738787009408
    MSE: 4357310.402080086
    RMSE: 2087.4171605311876



```python
# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))
```




    0.026887077025966846




```python
metrics.r2_score(y_test, y_pred)
```




    0.7979176783247053



Hyperparameter Tuning
Choose following method for hyperparameter tuning
RandomizedSearchCV --> Fast
GridSearchCV
Assign hyperparameters in form of dictionery
Fit the model
Check best paramters and best score


```python
from sklearn.model_selection import RandomizedSearchCV
```


```python
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
```


```python

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
```


```python
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
```


```python
rf_random.fit(X_train,y_train)
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    [CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   3.0s
    [CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    3.0s remaining:    0.0s


    [CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   3.2s
    [CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 
    [CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   3.1s
    [CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 
    [CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   3.1s
    [CV] n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10 
    [CV]  n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features=sqrt, max_depth=10, total=   3.1s
    [CV] n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   4.8s
    [CV] n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   4.8s
    [CV] n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   4.5s
    [CV] n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   4.5s
    [CV] n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=1100, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=15, total=   4.4s
    [CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   2.7s
    [CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   2.8s
    [CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   2.7s
    [CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   2.7s
    [CV] n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=100, min_samples_leaf=5, max_features=auto, max_depth=15, total=   2.8s
    [CV] n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   4.8s
    [CV] n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   5.0s
    [CV] n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   5.0s
    [CV] n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   5.1s
    [CV] n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15 
    [CV]  n_estimators=400, min_samples_split=5, min_samples_leaf=5, max_features=auto, max_depth=15, total=   5.0s
    [CV] n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20, total=   7.7s
    [CV] n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20, total=   7.6s
    [CV] n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20, total=   7.5s
    [CV] n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20, total=   7.5s
    [CV] n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=5, min_samples_leaf=10, max_features=auto, max_depth=20, total=   7.5s
    [CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25 
    [CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25, total=   7.0s
    [CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25 
    [CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25, total=   6.9s
    [CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25 
    [CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25, total=   7.1s
    [CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25 
    [CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25, total=   7.2s
    [CV] n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25 
    [CV]  n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, max_depth=25, total=   7.4s
    [CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   3.3s
    [CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   3.5s
    [CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   3.5s
    [CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   3.4s
    [CV] n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=1100, min_samples_split=15, min_samples_leaf=10, max_features=sqrt, max_depth=5, total=   3.2s
    [CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.3s
    [CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.7s
    [CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.4s
    [CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.4s
    [CV] n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15 
    [CV]  n_estimators=300, min_samples_split=15, min_samples_leaf=1, max_features=sqrt, max_depth=15, total=   1.4s
    [CV] n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.8s
    [CV] n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.7s
    [CV] n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.7s
    [CV] n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.7s
    [CV] n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5 
    [CV]  n_estimators=700, min_samples_split=10, min_samples_leaf=2, max_features=sqrt, max_depth=5, total=   1.7s
    [CV] n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20, total=   9.4s
    [CV] n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20, total=   9.3s
    [CV] n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20, total=   9.1s
    [CV] n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20, total=   9.2s
    [CV] n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20 
    [CV]  n_estimators=700, min_samples_split=15, min_samples_leaf=1, max_features=auto, max_depth=20, total=   9.7s


    [Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:  3.8min finished





    RandomizedSearchCV(cv=5, estimator=RandomForestRegressor(), n_jobs=1,
                       param_distributions={'max_depth': [5, 10, 15, 20, 25, 30],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 5, 10],
                                            'min_samples_split': [2, 5, 10, 15,
                                                                  100],
                                            'n_estimators': [100, 200, 300, 400,
                                                             500, 600, 700, 800,
                                                             900, 1000, 1100,
                                                             1200]},
                       random_state=42, scoring='neg_mean_squared_error',
                       verbose=2)




```python
rf_random.best_params_
```




    {'n_estimators': 700,
     'min_samples_split': 15,
     'min_samples_leaf': 1,
     'max_features': 'auto',
     'max_depth': 20}




```python

prediction = rf_random.predict(X_test)
```


```python
plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()
```


![png](output_72_0.png)



```python
plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
```


![png](output_73_0.png)



```python
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
```

    MAE: 1164.0603041304555
    MSE: 4045251.866085099
    RMSE: 2011.2811504325045



```python

```
