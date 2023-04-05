# CA-Projects-template-submission-repo
India has one of the fastest expanding economies in the world. Startups may be small businesses, but they can have a huge impact on economic growth. They generate more jobs, which leads to a healthier economy. Not only that, but startups can also contribute to economic vitality by encouraging innovation and injecting competition. 

## Summary
| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------:|------:|
| LP1 | Indian Startups Analysis|  [https://www.linkedin.com/posts/francis-acheampong_indian-startups-ecosystems-activity-7049292425093095424-08VA?utm_source=share&utm_medium=member_android](/) | [https://app.powerbi.com/groups/me/reports/be286258-d911-4dde-8fc9-b07411ec0251/ReportSectione65b9d28152b32fba38b](/) |

## Project Description
This project aims to provide insights into the Indian start-up ecosystem, including its current state, trends, and potential opportunities for growth. To achieve this, we will be analyzing key metrics in funding received by startups in India from 2018 to 2021. Through data analysis and visualization, we aim to identify key areas of focus for our team to enter the market and make a significant impact.

## Setup
# Data handling
import numpy as np 
import pandas as pd 
import glob

# Vizualisation (Matplotlib, Plotly, Seaborn, etc. )
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns 
sns.set_style('whitegrid')

# import plotly.express as px


from scipy import stats

from scipy.stats import pearsonr

from scipy.stats import chi2_contingency
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt

# Other packages
import os
import re
#display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings('ignore')

## App Execution
# importing 2018 data separate it has different form from the other ones
startup2018=pd.read_csv('startup_funding2018.csv',
                       usecols = ['Company Name', 'Industry', 'Round/Series', 'Amount','Location'])
# renaming columns for consistency
# Industry = sector 
# Round/Series = stage
startup2018.rename(columns ={'Industry':'Sector'}, inplace = True)
startup2018.rename(columns ={'Round/Series':'Stage'}, inplace = True)
startup2018.rename(columns ={'Amount':'Amount($)'}, inplace = True)
# adding new column funding year
startup2018['Funding Year']= "2018"

# Changing the founding year into integer
startup2018['Funding Year'] =startup2018['Funding Year'].astype(int)

# Taking  a deep look at the dataset across the years, certain columns were common in all the datasets and were deemed important in the analysis. so in importing the datasets, specific columns were imported. These included Company Name, Industry, Round/Series, Amount and Location


# To ensure consistency, some columns were renamed to match the columns in the dataset of subsequent years. Industry was renamed to Sector and Round/Series to Stage


# A new column named Funding Year was also added to each dataset and all rows were assigned the value 2018, depending on the dataset. 


# With the exception of 2018, all the other datasets had other relevant columns such as the year the startup was founded as well as the investor that provided funding. these columns were imported because there was a need to understand if the number of years that a startup had been
# DATA CLEANING 
The data cleaning phase follows the following trends

# Univariate Analysis
cleaning the Location clumns
# Maintaining only the first city 
startup2018['Location']=startup2018.Location.str.split(',').str[0]
startup2018['Location'].head()

# Here we have been able to remove all the cities after the comma
# Sector column

# Maintaining only the first sector
startup2018['Sector']=startup2018.Sector.str.split(',').str[0]
startup2018['Sector'].head()

startup2018.dropna(subset=['Amount($)'], inplace=True)

startup2018.loc[get_index,['Amount($)']] = pd.to_numeric(startup2018.loc[get_index,['Amount($)']].squeeze(), errors='coerce') * 0.012

startup2018.dropna(subset=['Amount($)'], inplace=True)

We realizing that the amount column is in object data type we needed to change to numeric datatype

# getting the index all rows in the column amount that has rupees
get_index=startup2018.index[startup2018['Amount($)'].str.contains('₹')]

# charging the rows in rupees to dollars using standard rate 

startup2018.loc[get_index,['Amount($)']]=startup2018.loc[get_index,['Amount($)']].values*0.012

startup2018.loc[:,['Amount($)']].head()

# Removing the symbols and commas from the amount column ₹, $, and ,
startup2018['Amount($)'] = startup2018['Amount($)'].apply(lambda x: str(x).replace('₹', ''))
startup2018['Amount($)'] = startup2018['Amount($)'].apply(lambda x: str(x).replace('$', ''))
startup2018['Amount($)'] = startup2018['Amount($)'].apply(lambda x: str(x).replace(',', ''))

Now we have agreed as a group that any amount without the synbol is considered as dollar 
hence all amount are in dollars

startup2018['Amount($)']=pd.to_numeric(startup2018['Amount($)'], errors='coerce')

startup2018.info()
# Dropping duplicate rows
startup2018= startup2018.drop_duplicates(keep='first')

startup2018.to_csv('cleaned_2018.csv')
# Replace the undisclosed values with np.nan
# Importing the 2019, 2020 and 2021 datasets

startup2018=pd.read_csv('cleaned_2018.csv')
startup2019=pd.read_csv('startup_funding2019.csv')
startup2020=pd.read_csv('startup_funding2020.csv')
startup2021=pd.read_csv('startup_funding2021.csv')

# Renaming columns to suit dataset in 2018
startup2019.rename(columns ={'Founded':'Funding Year'}, inplace = True)
startup2019.rename(columns ={'HeadQuarter':'Location'}, inplace = True)
startup2019.rename(columns ={'Company/Brand':'Company Name'}, inplace = True)

startup2020.rename(columns ={'Founded':'Funding Year'}, inplace = True)
startup2020.rename(columns ={'HeadQuarter':'Location'}, inplace = True)
startup2020.rename(columns ={'Company/Brand':'Company Name'}, inplace = True)


startup2021.rename(columns ={'Founded':'Funding Year'}, inplace = True)
startup2021.rename(columns ={'HeadQuarter':'Location'}, inplace = True)
startup2021.rename(columns ={'Company/Brand':'Company Name'}, inplace = True)

startup2019.to_csv('cleaned_2019.csv')
startup2020.to_csv('cleaned_2020.csv')
startup2021.to_csv('cleaned_2021.csv')

# importing 2018, 2019, 2020 and 2021 dataset together

csv_files = glob.glob('*.{}'.format('csv'))
csv_files

#Reading the data as DataFrame
data_final=pd.DataFrame()

for file in csv_files:
    df=pd.read_csv(file)

    data_final=data_final.append(df, ignore_index=True)

data_final.columns

data_final.drop(['Unnamed: 9'], axis=1, inplace=True)
data_final.drop(['Unnamed: 0'], axis=1, inplace=True)
data_final.drop(['What it does'], axis=1, inplace=True)
data_final.drop(['Unnamed: 0.1'], axis=1, inplace=True)

data_final.shape
data_final['Amount($)']=pd.to_numeric(data_final['Amount($)'], errors='coerce')

data_final['Funding Year'].unique()

data_final['Funding Year'] = data_final['Funding Year'].apply(lambda x: str(x).replace('-', ''))

data_final['Funding Year'] = data_final['Funding Year'].apply(lambda x: str(x).replace('nan', ''))

data_final['Funding Year']=pd.to_numeric(data_final['Funding Year'], errors='coerce')
#data_final['Funding Year']=data_final['Funding Year'].astype(int)


# Using the Simple Imputer method to replace missing values in the funding year column
imp=SimpleImputer(strategy='mean')
data_final['Funding Year']=imp.fit_transform(data_final['Funding Year'].values.reshape(-1,1))
data_final['Funding Year'].isna().sum()

data_final.describe().T

data_final['Funding Year'].unique()

missing_percentage=data_final.isna().mean()*100
missing_percentage

# Using the Simple Imputer method to replace missing values in the amount column

imp=SimpleImputer(strategy='median')
data_final['Amount($)']=imp.fit_transform(data_final['Amount($)'].values.reshape(-1,1))
data_final['Amount($)'].isna().sum()

missing_percentage=data_final.isna().mean()*100
missing_percentage

data_final['Stage'].unique()
# We remove the website url from the stage since it is not a stage name

data_final['Stage'] = data_final['Stage'].apply(lambda x: str(x).replace('https://docs.google.com/spreadsheets/d/1x9ziNeaz6auNChIHnMI8U6kS7knTr3byy_YBGfQaoUA/edit#gid=1861303593', 'Seed Funding'))

data_final['Stage'].unique()
# Removing and replacing the mismatch values for instance moneys appears in stage
data_final['Stage'] = data_final['Stage'].apply(lambda x: str(x).replace('$6000000', 'Seed Funding'))
data_final['Stage'] = data_final['Stage'].apply(lambda x: str(x).replace('$1000000', 'Seed Funding'))
data_final['Stage'] = data_final['Stage'].apply(lambda x: str(x).replace('$300000', 'Pre seed Round'))
data_final['Stage'] = data_final['Stage'].apply(lambda x: str(x).replace('$1200000', 'Series B3'))
data_final['Stage'] = data_final['Stage'].apply(lambda x: str(x).replace('nan', 'Bridge Round'))

missing_percentage=data_final.isna().mean()*100
missing_percentage

data_final['Investor'].unique()
data_final['Investor'] = data_final['Investor'].apply(lambda x: str(x).replace('nan', 'Undisclosed'))

missing_percentage=data_final.isna().mean()*100
missing_percentage
data_final['Founders'].unique()

data_final['Founders'] = data_final['Founders'].apply(lambda x: str(x).replace('nan', 'Arnav Kumar'))

missing_percentage=data_final.isna().mean()*100
missing_percentage

data_final['Location'].unique()
missing_percentage=data_final.isna().mean()*100
missing_percentage

## At the data preparation stage, we inspect the datasets in full, present it, test our hypotheses and rethink the cleaning, and creating new  features that will help us answer our question asked in the begining. 

# Dataset overview
# We inspected our final dataset using the following methods: .head(), .info(), .tail(), .shape()

index_ = data_final.index[data_final['Investor']=='Undisclosed']
index_

data_final['Investor'] = data_final['Investor'].replace('Undisclosed', np.nan)

#Strip the location data to only the city-area. 
data_final['Location'] = data_final.Location.str.split(',').str[0]
data_final['Location'].head()

plt.figure(figsize=(10,6))
sns.heatmap(data_final.isna(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})

data_final.isnull().sum()


# Find the mode of the 'Investor' column
mode = data_final['Founders'].mode()[0]
mode

# Replace missing values in the 'Founders' column with the mode
data_final['Founders'].fillna(mode, inplace=True)
# Find the mode of the 'Location' column
mode = data_final['Location'].mode()[0]

# Replace missing values in the 'Location' column with the mode
data_final['Location'].fillna(mode, inplace=True)

data_final.isnull().sum()

# drop the NaN values
data_final = data_final.dropna()

plt.figure(figsize=(10,6))
sns.heatmap(data_final.isna(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})

data_final.isnull().sum()

# Column by column analysis 

In this session, we discuss each column into detail to ascertain some level of information that assist us answer our question asked in week 1. This may include the use of .describe() method, matplotlib and other statistical tools

# we will first consider the Company Name

data_final['Company Name'].head()

data_final['Sector'].head()

data_final['Stage'].head()

# Amount($)

data_final['Amount($)'].head()

# calculate basic statistical measures
data_final_mean = data_final['Amount($)'].mean()
data_final_median = data_final['Amount($)'].median()
data_final_mode = data_final['Amount($)'].mode()
data_final_std_dev = data_final['Amount($)'].std()
data_final_min_val = data_final['Amount($)'].min()
data_final_max_val = data_final['Amount($)'].max()

print("Mean: ", data_final_mean)
print("Median: ", data_final_median)
print("Mode: ", data_final_mode)
print("Standard Deviation: ", data_final_std_dev)
print("Minimum Value: ", data_final_min_val)
print("Maximum Value: ", data_final_max_val)

# Analysis on the Sector

# Group the DataFrame by the 'Stage' column and count the occurrences of each stage
Investor_counts = data_final.groupby('Investor')['Amount($)'].count().reset_index()

# Sort the counts in descending order and select the top 5 values
Investor_counts = Investor_counts.sort_values(by='Amount($)', ascending=False).tail(10)

Investor_counts

corr_matrix=data_final.corr()

corr_matrix

sns.regplot(x='Amount($)', y='Funding Year', data=data_final, scatter_kws={'color':'red'}, line_kws={'color':'blue'})

x = data_final['Amount($)']
y = data_final['Funding Year']

corr, _ = pearsonr(x, y)
plt.scatter(x, y)
plt.xlabel('Amount($)')
plt.ylabel('Funding Year')
plt.title('Scatter plot with correlation coefficient')
plt.annotate(f'corr {corr:.2f}', (0.5, 0.5), xycoords='axes fraction', ha='center')
plt.show()


# QUESTIONS TO BE ANSWERED

# QUESTION 1: Which type of start-up location gets the most funding?

index_new = data_final.index[data_final['Location']=='California']
#index_new
Location_data = data_final.drop(labels=index_new, axis=0)

Location_grp = Location_data.groupby('Location')['Amount($)'].sum().reset_index()
top_10_locations = Location_grp.sort_values(by = 'Amount($)', ascending = False).head(10)
#top_5_locations = Location_grp.sort_values(by = 'Amount($)', ascending = False).head(5)
top_10_locations

fig = plt.figure(figsize = (10, 5))

# Create a bar chart to represent the answer

sns.barplot(x='Location', y='Amount($)', data=top_10_locations)

# Adding labels and title
plt.xlabel("Location")
plt.ylabel("Amount($)")
plt.title("Top 10 locations of startups with most funding")


#ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))


plt.show()
image.png


# From the visual above we can say that Bangalore recieved the most funding followed by Mumbai which had less than half of Bangalore's value. It can also be said that the bottom three startups that recieved the least amount are Delhi, Hyderbad and Gurgaon respectively

# QUESTION 2: At which stage do start-ups get more funding from investors?

stage_data = data_final.groupby('Stage')['Amount($)'].sum().reset_index()
top_10_stages = stage_data.sort_values(by = 'Amount($)', ascending = False).head(10)
top_10_stages

#displaying the results of the top 10 stages 

fig = plt.figure(figsize = (10, 5))

# Create a bar chart to represent our answer
sns.barplot(x='Stage', y='Amount($)', data=top_10_stages)

# Adding labels and title
plt.xlabel("Stage")
plt.ylabel("Amount")
plt.title("Top 10 startups who received the most funding")


#ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))


plt.show()

image.png


# Bridge Round stage recieved the most funding, Series D had the least funding. It can be concluded that Bridge Round recieved more than twice the funds recieved by Seed stage and any other stages.

# Question 3: Which type of investors invest the most money?

Investor_data = data_final.groupby('Investor')['Amount($)'].sum().reset_index()
Investor_data = Investor_data.sort_values(by = 'Amount($)', ascending = False)

#listing the top 10 stages

Investor_10_data = Investor_data.head(10)
Investor_10_data

fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Amount($)', y='Investor', data=Investor_10_data)

# Adding labels and title
plt.xlabel("Amount($)")
plt.ylabel("Investor")
plt.title("Top 10 Investors who invested the most")


# set y ticks and labels
plt.xticks(rotation = 0)

# Show the plot
plt.show()

[label](blob:vscode-webview%3A//1ojtcg0f9anrh3s8hprj7tj2regnjrfva6v5p9scugekiiqnvoa9/4d93a6fd-cf24-419b-903b-fa830d2e6df4)

# The bar chat above represents the share of how much each investor had invested. Inflection Point Venures topped the list, followed by Venture. Alteria Capital, Sequoia Capital India and Better Capital invested same amount.

# Question 4: Which type of investors invested the least money?

Investor_data = data_final.groupby('Investor')['Amount($)'].sum().reset_index()
Investor_data = Investor_data.sort_values(by = 'Amount($)', ascending = False)

#listing the least 5 investors who invested the least money

Investor_5_data = Investor_data.tail(5)
Investor_5_data

fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Amount($)', y='Investor', data=Investor_5_data)

# Adding labels and title
plt.xlabel("Amount($)")
plt.ylabel("Investor")
plt.title("The least five investors")


# set y ticks and labels
plt.xticks(rotation = 90)

# Show the plot
plt.show()

[label](blob:vscode-webview%3A//1ojtcg0f9anrh3s8hprj7tj2regnjrfva6v5p9scugekiiqnvoa9/803debb9-ae3b-419f-8812-5d2ce84a2be9)

# QUESTION 5: What is the percentage of Technology and Non-Technology in the Indian startups?

To answer this question we need to define and classify startups that belong to the group of technology and non technology

# Define the keywords
keywords = ["fintech", "edtech", "e-commerce","robotics", "cryptocurrency", "esports",
            "automotive ", "engineering ","telecommunications", "electricity", 
            "agritech", "healthtech", "technology", "e-marketplace", "social", 
            "tech", "gaming", "computer", "femtech", "solar", "embedded ", 
            "software ", "saas ", "e-commerce", "analytics", "ar", "vr", "crm", "nft", 
            "e-learning", "iot", "e-commerce", "e-mobility", "api ", 
            "ecommerce", "media", "ai","sportstech", "traveltech", "online", 
            "information", "automobile", "e-commerce", "biotechnology", "applications",  
            "it", "edtech", "energy", "computer", "agritech", "online ", "virtual ", 
            "fintech", "internet", "automation", "cloud", "apps", "chatbot", 
            "digital", "cleantech", "ev", "manufacturing","networking", "mobile ", 
            "electronics", "logitech", "solar", "insurtech","finance", "electric", 
            "fmcg", "intelligence", "blockchain","crypto", "foodtech ", "audio ", 
            "nanotechnology", "biometrics", "auto-tech", "biotech", "data ",  "autonomous ", 
            "AI", "machine learning", "e-market", "proptech", "machine learning "]

# the function here groups the keyword in the previous cell into technology and any other to non technology
    def check_keywords(string, keywords):
        for keyword in keywords:
            if keyword in string:
                return "technology"
        return "non-technology"

# Select only the rows with non-null values in the Sector column
data_final = data_final[data_final["Sector"].notnull()]

# Convert the Sector column to a Pandas Series
sector_series = pd.Series(data_final["Sector"])

#startup_funding_Full["Sector"].str.apply(check_keywords, keywords=keywords)

# Apply the check_keywords function to the Series
sector_series = sector_series.apply(check_keywords, keywords=keywords)

# Convert the resulting Series back to a column in the startup_funding_Full DataFrame
data_final["label"] = sector_series

This code prints the share of each label that is technology and non technology 

#Count the occurance of each unique term in the label column 

data_final["label"].value_counts(normalize=True)*100

From the output we can say that, Technology has the majority share compare to non technology

#A pie chart to show the distribution of the two labels 

plt.subplots(figsize = (10,8))
label = ['Technology ', 'Non-technology ']
label_data = data_final["label"].value_counts()

plt.pie(label_data, labels=label, autopct='%1.1f%%')

[label](blob:vscode-webview%3A//1ojtcg0f9anrh3s8hprj7tj2regnjrfva6v5p9scugekiiqnvoa9/0b67ac86-ef48-4c75-90e8-386495273f2d)


The pie chart represent the share these two groups. Technology has about 77% and Non-technology has about 22% as seen in the above pie chart

# Hypothesis Testing

NULL: Mumbai is the primary hub of Idian startups
Alternative: Mumbai is not the primary hub of Idian startups


To test this hypothesis, we created a grouby with location and company name. We grouped all location and number of startups in that location.

df_startups_per_city = data_final[['Location', 'Company Name']]
df_grouped_startups_per_city= df_startups_per_city.groupby('Location').count().reset_index()
df_grouped_startups_per_city

#.groupby('Investor')['Amount($)'].sum()

#Investor_data = data_final.groupby('Investor')['Amount($)'].sum().reset_index()
#Investor_data = Investor_data.sort_values(by = 'Amount($)', ascending = False)


df_grouped_startups_per_city.rename(columns={'Company Name':'Startup_counts'}, inplace = True)
df_grouped_startups_per_city


startup_top_10=df_grouped_startups_per_city.sort_values(by='Startup_counts',ascending=False)[:10]
startup_top_10.head(10)

#Investor_data = Investor_data.sort_values(by = 'Amount($)', ascending = False)


fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Location', y='Startup_counts', data=startup_top_10)

# Adding labels and title
plt.xlabel("Location")
plt.ylabel("Startup_counts")
plt.title("Top 10 city hubs")


# set y ticks and labels
plt.xticks(rotation = 0)

# Show the plot
plt.show()

[label](blob:vscode-webview%3A//1ojtcg0f9anrh3s8hprj7tj2regnjrfva6v5p9scugekiiqnvoa9/829c50a9-6ea2-4fb3-acfd-10a43492e067)

df_pie=startup_top_10.head()
df_pie.info()

df_pie=df_pie[:2]
plt.pie(df_pie['Startup_counts'], labels=df_pie['Location'], autopct='%1.1f%%')


plt.show()
#plt.pie(label_data, labels=label, autopct='%1.1f%%')


[label](blob:vscode-webview%3A//1ojtcg0f9anrh3s8hprj7tj2regnjrfva6v5p9scugekiiqnvoa9/fd200040-ae5e-48bb-a024-029824d48b01)

From the above charts, clearly it can been seen that Mumbai has fewer startup as compre to Bangalore. 
Infact Bangalore had twice more startups than what Mumbai had.
We therefore rejected the NULL hyposthesis. This is beacause there is enough evidence to do so.

Bangalore is the primary hub for startups in Indian.






## Author
ACHEAMPONG FRANCIS (TEAM LEAD)
CYCUS MURUMBA SITUMA 
HILDAH WANBUI
KERICH MIKE 
STEPHEN TETTEH OKOE 
...

