#%% md

## **Preamble**
Title: Beer Rater

Author: Kyle Weidner

Email: kweidne2@msudenver.edu

Last Update: 12/06/2021

#%% md

# **Introduction**
This project analyzes a dataset of 5500+ different beers, their characteristics, and their average ratings. The program
then runs Random Forest Regression to predict the rating of a beer based on its characteristics.
The program then attempts to find correlation between rating and beer characteristics such as ABV, Astringency, Body, etc.

#%% md

# **Dataset**
The dataset for this report was obtained from Kaggle
https://www.kaggle.com/stephenpolozoff/top-beer-information/version/3?select=beer_data_set.csv

This dataset consists of up to 50 top-rated beers across 112 styles, 5558 beers in total
The categories of the dataset are described as follows:

* <b>Name</b><br>
Beer's name

* <b>Key</b><br>
Beer's unique ID

* <b>Style</b><br>
The style of the beer. More information on beer-styles can be found here: https://www.craftbeer.com/beer/beer-styles-guide

* <b>Brewery</b><br>
The name of the beer's source

* <b>Description</b><br>
Notes on the beer (if available)

* <b>ABV</b><br>
Alcohol By Volume

* <b>Astringency</b><br>
Refers to the puckery or drying sensation created in the mouth and throat

* <b>Ave Rating</b><br>
The average rating of the beer at the time of collection

* <b>Min IBU</b><br>
The minimum IBU value each beer can possess. IBU was not a value available for each beer, but the IBU range for each style was.

* <b>Max IBU</b><br>
The maximum IBU value each beer can possess. IBU was not a value available for each beer, but the IBU range for each style was.

* <b>Body</b><br>
Refers to the weight or thickness of a beer

* <b>Alcohol</b><br>
I cannot find information about this category. I assume it measures the prevalence of the taste of alcohol

* <b>Bitter</b><br>
The bitterness level of the beer's taste

* <b>Sweet</b><br>
The sweetness level of the beer's taste

* <b>Sour</b><br>
The sourness level of the beer's taste

* <b>Salty</b><br>
The saltiness level of the beer's taste

* <b>Fruits</b><br>
The fruitiness level of the beer's taste

* <b>Hoppy</b><br>
The hoppiness level of the beer's taste

* <b>Spices</b><br>
The spices level of the beer's taste

* <b>Malty</b><br>
The maltiness level of the beer's taste

#%% md

# **Preprocessing**

The values are stored in a pandas dataframe and the beers that have null values in any columns are dropped.
A second dataframe is created, with unnecessary columns omitted, for the use of Random Forest Regression

#%%

# CS390Z - Introduction to Data Mining - Fall 2021
# Instructor: Thyago Mota
# Student: Kyle Weidner
# Description: Beer Rater Final Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.cbook import boxplot_stats

# Data Pre-processing & Cleaning
df = pd.read_csv('../data/beer_data_set.csv')
df = df.dropna(how='any', axis=0)  # drops any row with null data
df2 = df.drop(columns=["Name", "Style", "Brewery", "Description"])
df

#%% md

# **Summary Statistics**
A summary of the various beer characteristic statistics.

#%%

df.describe()

#%% md

# **Data Visualization**
A histogram visualization to represent each of the beer's characteristics.

#%%

# Data Visualization
df.hist(figsize=(15, 15))

#%% md

# **Outlier Analysis**
Each beer characteristic is analyzed and displayed in a boxplot. Each boxplot's outliers are listed with information
about the specific beer.

#%%

# # OUTLIER ANALYSIS VIA BOXPLOT
ave_rating_outliers = []
abv_outliers = []
min_ibu_outliers = []
max_ibu_outliers = []
astringency_outliers = []
body_outliers = []
alcohol_outliers = []
bitter_outliers = []
sweet_outliers = []
sour_outliers = []
salty_outliers = []
fruits_outliers = []
hoppy_outliers = []
spices_outliers = []
malty_outliers = []

#%% md

## Average Rating Outlier Analysis


#%%

# Average Rating
df2.boxplot(column=["Ave Rating"], grid=False)
stats = boxplot_stats(df2["Ave Rating"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Ave Rating"] > whisker_max:
        ave_rating_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Avg Rating: " + str(row["Ave Rating"]))
    if row["Ave Rating"] < whisker_min:
        ave_rating_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Avg Rating: " + str(row["Ave Rating"]))
for beer in ave_rating_outliers:
    print(beer)
plt.show()

#%% md

## ABV Outlier Analysis

#%%

# ABV
df2.boxplot(column=["ABV"], grid=False)
stats = boxplot_stats(df2["ABV"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["ABV"] > whisker_max:
        abv_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - ABV: " + str(row["ABV"]))
    if row["ABV"] < whisker_min:
        abv_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - ABV: " + str(row["ABV"]))
for beer in abv_outliers:
    print(beer)
plt.show()

#%% md

## Min IBU Outlier Analysis

#%%

# Min IBU
df2.boxplot(column=["Min IBU"], grid=False)
stats = boxplot_stats(df2["Min IBU"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Min IBU"] > whisker_max:
        min_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Min IBU: " + str(row["Min IBU"]))
    if row["ABV"] < whisker_min:
        min_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Min IBU: " + str(row["Min IBU"]))
for beer in min_ibu_outliers:
    print(beer)
plt.show()

#%% md

## Max IBU Outlier Analysis

#%%

# Max IBU
df2.boxplot(column=["Max IBU"], grid=False)
stats = boxplot_stats(df2["Max IBU"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Max IBU"] > whisker_max:
        max_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Max IBU: " + str(row["Max IBU"]))
    if row["Max IBU"] < whisker_min:
        max_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Max IBU: " + str(row["Max IBU"]))
for beer in max_ibu_outliers:
    print(beer)
plt.show()

#%% md

## Astringency Outlier Analysis

#%%

# Astringency
df2.boxplot(column=["Astringency"], grid=False)
stats = boxplot_stats(df2["Astringency"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Astringency"] > whisker_max:
        astringency_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Astringency: " + str(row["Astringency"]))
    if row["Astringency"] < whisker_min:
        astringency_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Astringency: " + str(row["Astringency"]))
for beer in astringency_outliers:
    print(beer)
plt.show()

#%% md

## Body Outlier Analysis

#%%

# Body
df2.boxplot(column=["Body"], grid=False)
stats = boxplot_stats(df2["Body"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Body"] > whisker_max:
        body_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Body: " + str(row["Body"]))
    if row["Body"] < whisker_min:
        body_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Body: " + str(row["Body"]))
for beer in body_outliers:
    print(beer)
plt.show()

#%% md

## Alcohol Outlier Analysis

#%%

# Alcohol
df2.boxplot(column=["Alcohol"], grid=False)
stats = boxplot_stats(df2["Alcohol"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Alcohol"] > whisker_max:
        alcohol_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Alcohol: " + str(row["Alcohol"]))
    if row["Alcohol"] < whisker_min:
        alcohol_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Alcohol: " + str(row["Alcohol"]))
for beer in alcohol_outliers:
    print(beer)
plt.show()

#%% md

## Bitter Outlier Analysis

#%%

# Bitter
df2.boxplot(column=["Bitter"], grid=False)
stats = boxplot_stats(df2["Bitter"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Bitter"] > whisker_max:
        bitter_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Bitter: " + str(row["Bitter"]))
    if row["Bitter"] < whisker_min:
        bitter_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Bitter: " + str(row["Bitter"]))
for beer in bitter_outliers:
    print(beer)
plt.show()

#%% md

## Sweet Outlier Analysis

#%%

# Sweet
df2.boxplot(column=["Sweet"], grid=False)
stats = boxplot_stats(df2["Sweet"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Sweet"] > whisker_max:
        sweet_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sweet: " + str(row["Sweet"]))
    if row["Sweet"] < whisker_min:
        sweet_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sweet: " + str(row["Sweet"]))
for beer in sweet_outliers:
    print(beer)
plt.show()

#%% md

## Sour Outlier Analysis

#%%

# Sour
df2.boxplot(column=["Sour"], grid=False)
stats = boxplot_stats(df2["Sour"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Sour"] > whisker_max:
        sour_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sour: " + str(row["Sour"]))
    if row["Sour"] < whisker_min:
        sour_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sour: " + str(row["Sour"]))
for beer in sour_outliers:
    print(beer)
plt.show()

#%% md

## Salty Outlier Analysis

#%%

# Salty
df2.boxplot(column=["Salty"], grid=False)
stats = boxplot_stats(df2["Salty"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Salty"] > whisker_max:
        salty_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Salty: " + str(row["Salty"]))
    if row["Salty"] < whisker_min:
        salty_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Salty: " + str(row["Salty"]))
for beer in salty_outliers:
    print(beer)
plt.show()

#%% md

## Fruits Outlier Analysis

#%%

# Fruits
df2.boxplot(column=["Fruits"], grid=False)
stats = boxplot_stats(df2["Fruits"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Fruits"] > whisker_max:
        fruits_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Fruits: " + str(row["Fruits"]))
    if row["Fruits"] < whisker_min:
        fruits_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Fruits: " + str(row["Fruits"]))
for beer in fruits_outliers:
    print(beer)
plt.show()

#%% md

## Hoppy Outlier Analysis

#%%

# Hoppy
df2.boxplot(column=["Hoppy"], grid=False)
stats = boxplot_stats(df2["Hoppy"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Hoppy"] > whisker_max:
        hoppy_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Hoppy: " + str(row["Hoppy"]))
    if row["Hoppy"] < whisker_min:
        hoppy_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Hoppy: " + str(row["Hoppy"]))
for beer in hoppy_outliers:
    print(beer)
plt.show()

#%% md

## Spices Outlier Analysis

#%%

# Spices
df2.boxplot(column=["Spices"], grid=False)
stats = boxplot_stats(df2["Spices"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Spices"] > whisker_max:
        spices_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Spices: " + str(row["Spices"]))
    if row["Spices"] < whisker_min:
        spices_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Spices: " + str(row["Spices"]))
for beer in spices_outliers:
    print(beer)
plt.show()

#%% md

## Malty Outlier Analysis

#%%

# Malty
df2.boxplot(column=["Malty"], grid=False)
stats = boxplot_stats(df2["Malty"].values)
whisker_max = stats[0]["whishi"]
whisker_min = stats[0]["whislo"]
for row in df.iloc:
    if row["Malty"] > whisker_max:
        malty_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Malty: " + str(row["Malty"]))
    if row["Malty"] < whisker_min:
        malty_outliers.append(
            "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Malty: " + str(row["Malty"]))
for beer in malty_outliers:
    print(beer)
plt.show()

#%% md

# **Correlation Analysis**
I attempt to find any correlation between Average Rating and the beer's characteristics. The variation of these
characteristics between beers is what makes up the beer's unique taste profile. It appears there is a slight
correlation between Average Rating and ABV, Min IBU, Max, IBU, Body, Sweet, Sour, and Fruits characteristics.

#%%

corr = df["Ave Rating"].corr(df["ABV"])
print("Rating / ABV Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Astringency"])
print("Rating / Astringency Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Min IBU"])
print("Rating / Min IBU Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Max IBU"])
print("Rating / Max IBU Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Body"])
print("Rating / Body Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Alcohol"])
print("Rating / Alcohol Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Bitter"])
print("Rating / Bitter Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Sweet"])
print("Rating / Sweet Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Sour"])
print("Rating / Sour Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Salty"])
print("Rating / Salty Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Fruits"])
print("Rating / Fruits Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Hoppy"])
print("Rating / Hoppy Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Spices"])
print("Rating / Spices Correlation: ", corr)
corr = df["Ave Rating"].corr(df["Malty"])
print("Rating / Malty Correlation: ", corr)

#%% md

# **Regression**
I run a Random Forest Regression to predict the beer's Average Rating based on its characteristics. This algorithm
analyzes the characteristics of a beer and then predicts the beer's rating with an accuracy of 94.65%.

#%%

# Random Forest Regression to Predict Avg Rating
X = df2.iloc[:, df2.columns != 'Ave Rating'].values
Y = df2.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
model = RandomForestRegressor(n_estimators=1000, random_state=1234)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
ap_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(ap_df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Calculate the errors
errors = abs(y_pred - y_test)
percentage_error = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(percentage_error)
print('Accuracy:', round(accuracy, 2), '%.')

#%% md

## **Conclusion**

By using a dataset containing information on 5500+ beers, I have shown correlation between various beer characteristics.
 I have used the data to predict what rating a beer would receive, depending on the beer's characteristics. Outlier
 Analysis was performed on each of the beer's characteristics. Given the characteristics of a beer, I was able to
 successfully predict a beer's rating with an accuracy of 94.6% by using Random Forest Regression.
