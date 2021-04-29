#!/usr/bin/env python
# coding: utf-8

# # Springboard Data Science Career Track Unit 4 Challenge - Tier 3 Complete
# 
# ## Objectives
# Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time. 
# 
# In this notebook, you're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the *slightly* messier work that data scientists do with actual datasets!
# 
# Here’s the mystery we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***
# 
# 
# A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London [(here's some info for the curious)](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.
# 
# ***This is the Tier 3 notebook, which means it's not filled in at all: we'll just give you the skeleton of a project, the brief and the data. It's up to you to play around with it and see what you can find out! Good luck! If you struggle, feel free to look at easier tiers for help; but try to dip in and out of them, as the more independent work you do, the better it is for your learning!***
# 
# This challenge will make use of only what you learned in the following DataCamp courses: 
# - Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
# - Data Types for Data Science
# - Python Data Science Toolbox (Part One) 
# - pandas Foundations
# - Manipulating DataFrames with pandas
# - Merging DataFrames with pandas
# 
# Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following: 
# - **pandas**
#     - **data ingestion and inspection** (pandas Foundations, Module One) 
#     - **exploratory data analysis** (pandas Foundations, Module Two)
#     - **tidying and cleaning** (Manipulating DataFrames with pandas, Module Three) 
#     - **transforming DataFrames** (Manipulating DataFrames with pandas, Module One)
#     - **subsetting DataFrames with lists** (Manipulating DataFrames with pandas, Module One) 
#     - **filtering DataFrames** (Manipulating DataFrames with pandas, Module One) 
#     - **grouping data** (Manipulating DataFrames with pandas, Module Four) 
#     - **melting data** (Manipulating DataFrames with pandas, Module Three) 
#     - **advanced indexing** (Manipulating DataFrames with pandas, Module Four) 
# - **matplotlib** (Intermediate Python for Data Science, Module One)
# - **fundamental data types** (Data Types for Data Science, Module One) 
# - **dictionaries** (Intermediate Python for Data Science, Module Two)
# - **handling dates and times** (Data Types for Data Science, Module Four)
# - **function definition** (Python Data Science Toolbox - Part One, Module One)
# - **default arguments, variable length, and scope** (Python Data Science Toolbox - Part One, Module Two) 
# - **lambda functions and error handling** (Python Data Science Toolbox - Part One, Module Four) 

# ## The Data Science Pipeline
# 
# This is Tier Three, so we'll get you started. But after that, it's all in your hands! When you feel done with your investigations, look back over what you've accomplished, and prepare a quick presentation of your findings for the next mentor meeting. 
# 
# Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as  [David Spiegelhalter](https://www.youtube.com/watch?v=oUs1uvsz0Ok) reminds us, there is no substitute for simply **taking a really, really good look at the data.** Sometimes, this is all we need to answer our question.
# 
# Data Science projects generally adhere to the four stages of Data Science Pipeline:
# 1. Sourcing and loading 
# 2. Cleaning, transforming, and visualizing 
# 3. Modeling 
# 4. Evaluating and concluding 
# 

# ### 1. Sourcing and Loading 
# 
# Any Data Science project kicks off by importing  ***pandas***. The documentation of this wonderful library can be found [here](https://pandas.pydata.org/). As you've seen, pandas is conveniently connected to the [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) libraries. 
# 
# ***Hint:*** This part of the data science pipeline will test those skills you acquired in the pandas Foundations course, Module One. 

# #### 1.1. Importing Libraries

# In[1]:


# Let's import the pandas, numpy libraries as pd, and np respectively. 
import pandas as pd
import numpy as np

# Load the pyplot collection of functions from matplotlib, as plt 
import matplotlib.pyplot as plt


# #### 1.2.  Loading the data
# Your data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal for London-oriented datasets. 

# In[2]:


# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices= "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)


# ### 2. Cleaning, transforming, and visualizing
# This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.
# 
# The end goal of data cleaning is to have tidy data. When data is tidy: 
# 
# 1. Each variable has a column.
# 2. Each observation forms a row.
# 
# Keep the end goal in mind as you move through this process, every step will take you closer. 
# 
# 
# 
# ***Hint:*** This part of the data science pipeline should test those skills you acquired in: 
# - Intermediate Python for data science, all modules.
# - pandas Foundations, all modules. 
# - Manipulating DataFrames with pandas, all modules.
# - Data Types for Data Science, Module Four.
# - Python Data Science Toolbox - Part One, all modules

# **2.1. Exploring your data** 
# 
# Think about your pandas functions for checking out a dataframe. 

# In[3]:


properties.shape
properties.head()


# **2.2. Cleaning the data**
# 
# You might find you need to transpose your dataframe, check out what its row indexes are, and reset the index. You  also might find you need to assign the values of the first row to your column headings  . (Hint: recall the .columns feature of DataFrames, as well as the iloc[] method).
# 
# Don't be afraid to use StackOverflow for help  with this.

# In[4]:


propertiesT = properties.transpose()
propertiesT.head()


# In[5]:


# Reset the index 
propertiesT = propertiesT.reset_index()

# Assign the values of the first row to the column headings
new_header = propertiesT.iloc[0]
propertiesT = propertiesT[1:]
propertiesT.columns = new_header

propertiesT.head()


# **2.3. Cleaning the data (part 2)**
# 
# You might we have to **rename** a couple columns. How do you do this? The clue's pretty bold...

# In[6]:


# Rename the first two columns to London_Borough and ID,respectively
propertiesT.rename(columns={'Unnamed: 0':'London_Borough',pd.NaT: 'ID'},inplace=True)


# **2.4.Transforming the data**
# 
# Remember what Wes McKinney said about tidy data? 
# 
# You might need to **melt** your DataFrame here. 

# In[7]:


# Melt the dataframe keeping London_Borough and ID as identifier variables
propertiesClean = pd.melt(propertiesT, id_vars = ['London_Borough','ID'])


# Remember to make sure your column data types are all correct. Average prices, for example, should be floating point numbers... 

# In[8]:


propertiesClean.head()


# In[9]:


# Rename the last two columns to Month and Average_Price, respectively
propertiesClean.rename(columns={0:'Month','value': 'Average_Price'},inplace=True)

propertiesClean.head()


# In[10]:


# Check data types in dataframe
propertiesClean.dtypes

# Changing Average Price to float
propertiesClean['Average_Price'] = pd.to_numeric(propertiesClean['Average_Price'])

#Rechecking data types
propertiesClean.dtypes


# **2.5. Cleaning the data (part 3)**
# 
# Do we have an equal number of observations in the ID, Average Price, Month, and London Borough columns? Remember that there are only 32 London Boroughs. How many entries do you have in that column? 
# 
# Check out the contents of the London Borough column, and if you find null values, get rid of them however you see fit. 

# In[11]:


propertiesClean.info()


# In[12]:


# There are only 13860 IDs and Average_Prices but 14784 London_Boroughs. Let's call the unique method
# on London_Boroughs to ensure there are only 32 unique boroughs.
propertiesClean['London_Borough'].unique()


# In[13]:


# There are several 'Unnamed: #' rows as well as a bunch of non-boroughs. First drop the 'Unnamed: #' rows.
propertiesClean = propertiesClean[(propertiesClean.London_Borough != 'Unnamed: 34') &                                   (propertiesClean.London_Borough != 'Unnamed: 37') &                                   (propertiesClean.London_Borough != 'Unnamed: 47')]
# Ensure there are 32 boroughs
propertiesClean['London_Borough'].nunique()


# In[14]:


# Now make a list of the non-boroughs
# Note: according to Google Search, 'City of London' is a principal division but not a London borough
nonBoroughs = ['City of London','Inner London','Outer London','NORTH EAST', 'NORTH WEST','YORKS & THE HUMBER',               'EAST MIDLANDS','WEST MIDLANDS','EAST OF ENGLAND','LONDON','SOUTH EAST','SOUTH WEST','England']


# In[15]:


# Eliminate rows in propertiesClean dataframe that are in nonBoroughs
propertiesClean = propertiesClean[~propertiesClean['London_Borough'].isin(nonBoroughs)]


# In[16]:


# Ensure there are 32 boroughs
propertiesClean['London_Borough'].nunique()


# In[17]:


# Look for null values in ID column
propertiesClean['ID'].isnull().values.any()


# In[18]:


# Look for null values in Average_Price column
propertiesClean['Average_Price'].isnull().values.any()


# In[19]:


# Final dataframe ready for visualization: df
df = propertiesClean


# **2.6. Visualizing the data**
# 
# To visualize the data, why not subset on a particular London Borough? Maybe do a line plot of Month against Average Price?

# In[20]:


# Visualize the average prices for Barnet (as an example, could have taken any)
barnet_prices = df[df['London_Borough']=='Barnet']
barnet_vis = barnet_prices.plot(kind='line',x='Month',y='Average_Price')
barnet_vis.set_xlabel('Month')
barnet_vis.set_ylabel('Average Price')
barnet_vis.legend(['Barnet Borough'])


# To limit the number of data points you have, you might want to extract the year from every month value your *Month* column. 
# 
# To this end, you *could* apply a ***lambda function***. Your logic could work as follows:
# 1. look through the `Month` column
# 2. extract the year from each individual value in that column 
# 3. store that corresponding year as separate column. 
# 
# Whether you go ahead with this is up to you. Just so long as you answer our initial brief: which boroughs of London have seen the greatest house price increase, on average, over the past two decades? 

# In[21]:


# Extract the year from the month column and add Year as a new column
df['Year'] = df['Month'].apply(lambda x: x.year)

# Call the tail() method on df
df.tail()


# **3. Modeling**
# 
# Consider creating a function that will calculate a ratio of house prices, comparing the price of a house in 2018 to the price in 1998.
# 
# Consider calling this function create_price_ratio.
# 
# You'd want this function to:
# 1. Take a filter of dfg, specifically where this filter constrains the London_Borough, as an argument. For example, one admissible argument should be: dfg[dfg['London_Borough']=='Camden'].
# 2. Get the Average Price for that Borough, for the years 1998 and 2018.
# 4. Calculate the ratio of the Average Price for 1998 divided by the Average Price for 2018.
# 5. Return that ratio.
# 
# Once you've written this function, you ultimately want to use it to iterate through all the unique London_Boroughs and work out the ratio capturing the difference of house prices between 1998 and 2018.
# 
# Bear in mind: you don't have to write a function like this if you don't want to. If you can solve the brief otherwise, then great! 
# 
# ***Hint***: This section should test the skills you acquired in:
# - Python Data Science Toolbox - Part One, all modules

# In[22]:


# Calculate the mean house price for each year for each borough
df_grouped = df.groupby(by=['London_Borough','Year']).mean()
df_grouped.head(10)


# In[23]:


# Reset the index
df_grouped = df_grouped.reset_index()
df_grouped.head(10)


# In[25]:


# Create a function to calculate the ratio of the average price in 2018 divided by the average price in 1998

def create_price_ratio(f):
    avg_price_1998 = float(f.loc[f['Year']==1998, 'Average_Price'])
    avg_price_2018 = float(f.loc[f['Year']==2018, 'Average_Price'])
    return [avg_price_2018/avg_price_1998]


# In[26]:


# Create a dictionary of the ratio for each unique London Borough
ratios = {}
for borough in df_grouped['London_Borough'].unique():
    ratios[borough] = create_price_ratio(df_grouped[df_grouped['London_Borough']==borough])


# In[27]:


# Convert dictionary to dataframe
df_ratios = pd.DataFrame(ratios)
df_ratios.head()


# In[28]:


# Transpose and reset index
df_ratiosT = df_ratios.transpose()
df_ratiosT = df_ratiosT.reset_index()
df_ratiosT.head()


# In[29]:


# Rename the columns
df_ratiosT.rename(columns={'index':'London_Borough',0: 'Housing_Ratio'},inplace=True)
df_ratiosT.head()


# In[41]:


# Let's sort in descending order and select the top 15 boroughs.
# Make a variable called top15, and assign it the result of calling sort_values() on df_ratios. 
top15= df_ratiosT.sort_values(by='Housing_Ratio',ascending=False).head(15)
print(top15)


# In[42]:


# Let's plot the boroughs that have seen the greatest changes in price.
# Make a variable called plotting Assign it the result of filtering top15 on 'London_Borough' and 'Housing_Ratio', then calling plot(), with
# the parameter kind = 'bar'. 
plotting = top15[['London_Borough','Housing_Ratio']].plot(kind='bar')

plotting.set_xticklabels(top15.London_Borough)


# ### 4. Conclusion
# What can you conclude? Type out your conclusion below. 
# 
# Look back at your notebook. Think about how you might summarize what you have done, and prepare a quick presentation on it to your mentor at your next meeting. 
# 
# We hope you enjoyed this practical project. It should have consolidated your data hygiene and pandas skills by looking at a real-world problem involving just the kind of dataset you might encounter as a budding data scientist. Congratulations, and looking forward to seeing you at the next step in the course! 

# In[ ]:


# The top 5 boroughs that experienced the greatest increase in housing prices on average are Hackney,
# Waltham Forest, Southwark, Lewisham, and Westminster.

#Hackney:6.198286
#Waltham Forest:5.834756
#Southwark:5.516485
#Lewisham:5.449221
#Westminster:5.353565

#Summary of data analysis:
#Read an excel file from an URL, which is a very crucial step in data analytics as we often need to get the data directly from the server in case the data is updated periodically.
#Then, converted the data into a tidy set of data where each borough and year has its corresponding average price value. In this process of tidying up the data, I had to do a lot of preprocessing including, but not limited to, filtering out non-meaningful entries, removing any null values, aligning the data into right shape (by performing transposing, melting, groupby etc.).
#Plotted the historical average price for all the boroughs to get an overall idea of the price range of different boroughs across the time. Since the data is plotted with the same y limit, we can deduce the relative priciness of different boroughs, such as City of London, Kensington & Chelsea, and Westminister are the most expensive boroughs, as expected as they are well-known center for business and social activities, to live in. However, that does not mean the relative price change compare to 1995 (where now is 2020) is also the highest in these boroughs. Some less/medium expensive boroughs can become relatively very expensive in comparison to 1995.
#Hackney, Waltham Forest (which are moderate to less expensive boroughs) was found to have the largest relative change in price which can mean multiple things: 1. they are being developed quickly, 2. maybe they are close to other expensive boroughs as a result people working in expensive boroughs are moving there to live etc.
#In conclusion, the findings would be tremendously valuable for a lot of businesses, including real estate, tech industries etc, to determine their business policies such as where to focus on for growth or where to shift to have low operational cost etc.
