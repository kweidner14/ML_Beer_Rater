## **Preamble**
Title: Beer Rater

Author: Kyle Weidner

Email: kweidne2@msudenver.edu

Last Update: 12/06/2021

# **Introduction**
This project analyzes a dataset of 5500+ different beers, their characteristics, and their average ratings. The program
then runs Random Forest Regression to predict the rating of a beer based on its characteristics.
The program then attempts to find correlation between rating and beer characteristics such as ABV, Astringency, Body, etc.

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
