#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
from numpy import cov
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import seaborn as sns

# Import the data set

tests = pd.read_csv (r'C:\Users\HTC\Desktop\FinalModel\yield3.csv')
print(tests)
print(tests.columns)


# In[ ]:



#check if the are any missing values in dataset entries, true if yes and false if no
print (tests.isnull())

#check sum of missing values for each dataset column
print(tests.isnull().sum())


# In[ ]:


#visuallize missing values using heatmap
sns.heatmap(tests.isnull(), cbar=False)
plt.show()


# In[ ]:


#drop year of production since its not important
tests.drop(tests.columns[[2]], axis = 1, inplace = True)
print(tests.columns)
#drop all missing or null values in the dataset
tests.dropna(inplace=True)
print(tests)


# In[ ]:


#verify using heatmap diagram that all missing values had been removed
sns.heatmap(tests.isnull(), cbar=False)
plt.show() 
#verify that sum of missing values is 0 for each dataset column after cleaning
print(tests.isnull().sum()) 


# In[ ]:


#Statistical calculations(finding mean, standard deviation, quartiles, minimum and maximum values for each column)
print(tests.shape)
print(tests.info())
print(tests.describe())

#data type for each dataset column
print()
dataTypeSeries = tests.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#Computing quartiles of data and outlier determination
Q1 = tests.quantile(0.25)
Q3 = tests.quantile(0.75)
IQR = Q3 - Q1
print(tests < (Q1 - 1.5 * IQR)) or (tests > (Q3 + 1.5 * IQR))


# In[ ]:


plt.boxplot(tests["Total_production"])
plt.title("Total_production box plot")
plt.show()

plt.boxplot(tests["Rainfall"])
plt.title("Rainfall box plot")
plt.show()

plt.boxplot(tests["Humidity"])
plt.title("Humidity box plot")
plt.show()

plt.boxplot(tests["Temperature"])
plt.title("Temperature box plot")
plt.show()

plt.boxplot(tests["Pesticides"])
plt.title("Pesticides box plot")
plt.show()

plt.boxplot(tests["Soil_ph"])
plt.title("Soil PH box plot")
plt.show()

plt.boxplot(tests["N"])
plt.title("Nitrogen Input box plot")
plt.show()

plt.boxplot(tests["P"])
plt.title("Phosphorus Input box plot")
plt.show()
                  
plt.boxplot(tests["K"])
plt.title("Potassium Input box plot")
plt.show()                 

plt.boxplot(tests["Area_Planted"])
plt.title("Area planted box plot")
plt.show()


# In[ ]:


import seaborn as sns
import pandas as pd
#Visuallization using count plot compound histogram based on categorical data
plt.title("Total Production Per Place Count Plot")
sns.countplot(x='Total_production', hue='Place', data=tests)
plt.show()

plt.title("Total Production Per Crop Name Count Plot")
sns.countplot(x='Total_production', hue='Crop_Name', data=tests)
plt.show()


# In[ ]:



import pandas as pd
#Converting Catergoric column data to numeric using one hot encoder
#Crop Name and Place dataset columns contained catergoric variables
crop_data = pd.get_dummies(tests['Crop_Name'], drop_first = True)
print(crop_data)
place_data = pd.get_dummies(tests['Place'], drop_first = True)
print(place_data)


# In[ ]:


#Concatinating catergorically converted numeric data with the original dataset
tests = pd.concat([tests, crop_data,place_data], axis = 1)
print(tests)

#Dropping older Crop Name and Place dataset columns since they are now unecessary
tests.drop(['Crop_Name', 'Place'], axis = 1, inplace = True)

#Printing new columns and data for the finally processed dataset
#print finally processed dataset
print('Finally Processed Dataset....')
print(tests.columns)
print(tests)


# In[ ]:



import matplotlib.pyplot as plt
#Visuallization using histogram
plt.title("Total Production Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Total Production Range')
plt.hist(tests['Total_production'].dropna())
plt.show()


plt.title("Area Planted Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Area Planted Range')
plt.hist(tests['Area_Planted'].dropna())
plt.show()

plt.title("Rainfall Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Rainfall Range')
plt.hist(tests['Rainfall'].dropna())
plt.show()


plt.title("Humidity Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Humidity Range')
plt.hist(tests['Humidity'].dropna())
plt.show()

plt.title("Temperature Readings Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Temperature Readings Range')
plt.hist(tests['Temperature'].dropna())
plt.show()


plt.title("Pesticides Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Pesticides Input Range')
plt.hist(tests['Pesticides'].dropna())
plt.show()

plt.title("Soil pH Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Soil pH Range')
plt.hist(tests['Soil_ph'].dropna())
plt.show()

plt.title("Nitrogen Input Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Nitrogen Input Range')
plt.hist(tests['N'].dropna())
plt.show()

plt.title("Phosphorus Input Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Phosphorus Input Range')
plt.hist(tests['P'].dropna())
plt.show()

plt.title("Potassium Input Histogram Data")
plt.ylabel('Frequency')
plt.xlabel('Potassium Input Range')
plt.hist(tests['K'].dropna())
plt.show()


# In[ ]:



import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
#Visuallisation using pie chart
tests = pd.read_csv (r'C:\Users\HTC\Desktop\FinalModel\yield3.csv')
tests.dropna(inplace=True)

sums = tests.groupby(tests["Place"])["Total_production"].sum()
plt.axes().set_aspect("equal")
plt.pie(sums, labels=sums.index)
plt.title("Total Production Per Place Pie Chart")
plt.show()

sums = tests.groupby(tests["Crop_Name"])["Total_production"].sum()
plt.axes().set_aspect("equal")
plt.pie(sums, labels=sums.index)
plt.title("Total Production Per Crop Name Pie Chart")
plt.show()


# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

tests = pd.read_csv (r'C:\Users\HTC\Desktop\FinalModel\yield3.csv')
tests.dropna(inplace=True)

#Coss correlation heatmap matrix
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(tests.corr(),annot = True)
plt.show()

#Visuallization using scatter plots
y_data = tests['Rainfall']

x_data = tests['Humidity']
print(x_data.shape)
print(y_data.shape)

plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Rainfall Vs Humidity Scatter plot')
plt.ylabel('Rainfall(mm) ')
plt.xlabel('Humidity(%) ')
plt.show()


y_data = tests['Rainfall']

x_data = tests['Temperature']
print(x_data.shape)
print(y_data.shape)

plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Rainfall Vs Temperature Scatter plot')
plt.ylabel('Rainfall(mm) ')
plt.xlabel('Temperature(deg C) ')
plt.show()


y_data = tests['Humidity']
x_data = tests['Temperature']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Humidity Vs Temperature Scatter plot')
plt.ylabel('Humidity(%) ')
plt.xlabel('Temperature(deg C) ')
plt.show()

y_data = tests['Total_production']
x_data = tests['Area_Planted']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Area Planted Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Area Planted(hectares) ')
plt.show()

y_data = tests['Total_production']
x_data = tests['Rainfall']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Rainfall Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Rainfall(mm) ')
plt.show()

y_data = tests['Total_production']
x_data = tests['Humidity']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Humidity Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Humidity(%) ')
plt.show()

y_data = tests['Total_production']
x_data = tests['Temperature']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Temperature Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Temperature(deg C) ')
plt.show()

y_data = tests['Total_production']
x_data = tests['Pesticides']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Pesticides Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Pesticides ')
plt.show()


y_data = tests['Total_production']
x_data = tests['Soil_ph']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Soil pH Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Soil pH')
plt.show()

y_data = tests['Total_production']
x_data = tests['N']
plt.scatter(x_data, y_data)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Total Production Vs Nitrogen Content Scatter plot')
plt.ylabel('Total Production(tonnes) ')
plt.xlabel('Nitrogen Content')
plt.show()

