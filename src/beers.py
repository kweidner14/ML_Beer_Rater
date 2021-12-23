# CS390Z - Introduction to Data Mining - Fall 2021
# Instructor: Thyago Mota
# Student: Kyle Weidner
# Description: Final Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.cbook import boxplot_stats

if __name__ == "__main__":
    # Data Pre-processing & Cleaning
    df = pd.read_csv('../data/beer_data_set.csv')
    df = df.dropna(how='any', axis=0)  # drops any row with null data

    # Summary Statistics
    print(df.describe())

    # Data Visualization
    df.hist(figsize=(15, 15))
    plt.show()


    # Random Forest Regression to Predict Avg Rating
    df2 = df.drop(columns=["Name", "Style", "Brewery", "Description"])
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

    # Correctly predicts the rating of a beer with 94.65% Accuracy!



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

    # # OUTLIER ANALYSIS VIA BOXPLOT
    # # Average Rating
    # df2.boxplot(column=["Ave Rating"], grid=False)
    # stats = boxplot_stats(df2["Ave Rating"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Ave Rating"] > whisker_max:
    #         ave_rating_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Avg Rating: " + str(row["Ave Rating"]))
    #     if row["Ave Rating"] < whisker_min:
    #         ave_rating_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Avg Rating: " + str(row["Ave Rating"]))
    # for beer in ave_rating_outliers:
    #     print(beer)
    # plt.show()
    #
    # # ABV
    # df2.boxplot(column=["ABV"], grid=False)
    # stats = boxplot_stats(df2["ABV"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["ABV"] > whisker_max:
    #         abv_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - ABV: " + str(row["ABV"]))
    #     if row["ABV"] < whisker_min:
    #         abv_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - ABV: " + str(row["ABV"]))
    # for beer in abv_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Min IBU
    # df2.boxplot(column=["Min IBU"], grid=False)
    # stats = boxplot_stats(df2["Min IBU"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Min IBU"] > whisker_max:
    #         min_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Min IBU: " + str(row["Min IBU"]))
    #     if row["ABV"] < whisker_min:
    #         min_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Min IBU: " + str(row["Min IBU"]))
    # for beer in min_ibu_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Max IBU
    # df2.boxplot(column=["Max IBU"], grid=False)
    # stats = boxplot_stats(df2["Max IBU"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Max IBU"] > whisker_max:
    #         max_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Max IBU: " + str(row["Max IBU"]))
    #     if row["Max IBU"] < whisker_min:
    #         max_ibu_outliers.append("'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Max IBU: " + str(row["Max IBU"]))
    # for beer in max_ibu_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Astringency
    # df2.boxplot(column=["Astringency"], grid=False)
    # stats = boxplot_stats(df2["Astringency"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Astringency"] > whisker_max:
    #         astringency_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Astringency: " + str(row["Astringency"]))
    #     if row["Astringency"] < whisker_min:
    #         astringency_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Astringency: " + str(row["Astringency"]))
    # for beer in astringency_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Body
    # df2.boxplot(column=["Body"], grid=False)
    # stats = boxplot_stats(df2["Body"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Body"] > whisker_max:
    #         body_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Body: " + str(row["Body"]))
    #     if row["Body"] < whisker_min:
    #         body_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Body: " + str(row["Body"]))
    # for beer in body_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Alcohol
    # df2.boxplot(column=["Alcohol"], grid=False)
    # stats = boxplot_stats(df2["Alcohol"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Alcohol"] > whisker_max:
    #         alcohol_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Alcohol: " + str(row["Alcohol"]))
    #     if row["Alcohol"] < whisker_min:
    #         alcohol_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Alcohol: " + str(row["Alcohol"]))
    # for beer in alcohol_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Bitter
    # df2.boxplot(column=["Bitter"], grid=False)
    # stats = boxplot_stats(df2["Bitter"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Bitter"] > whisker_max:
    #         bitter_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Bitter: " + str(row["Bitter"]))
    #     if row["Bitter"] < whisker_min:
    #         bitter_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Bitter: " + str(row["Bitter"]))
    # for beer in bitter_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Sweet
    # df2.boxplot(column=["Sweet"], grid=False)
    # stats = boxplot_stats(df2["Sweet"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Sweet"] > whisker_max:
    #         sweet_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sweet: " + str(row["Sweet"]))
    #     if row["Sweet"] < whisker_min:
    #         sweet_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sweet: " + str(row["Sweet"]))
    # for beer in sweet_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Sour
    # df2.boxplot(column=["Sour"], grid=False)
    # stats = boxplot_stats(df2["Sour"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Sour"] > whisker_max:
    #         sour_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sour: " + str(row["Sour"]))
    #     if row["Sour"] < whisker_min:
    #         sour_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Sour: " + str(row["Sour"]))
    # for beer in sour_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Salty
    # df2.boxplot(column=["Salty"], grid=False)
    # stats = boxplot_stats(df2["Salty"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Salty"] > whisker_max:
    #         salty_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Salty: " + str(row["Salty"]))
    #     if row["Salty"] < whisker_min:
    #         salty_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Salty: " + str(row["Salty"]))
    # for beer in salty_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Fruits
    # df2.boxplot(column=["Fruits"], grid=False)
    # stats = boxplot_stats(df2["Fruits"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Fruits"] > whisker_max:
    #         fruits_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Fruits: " + str(row["Fruits"]))
    #     if row["Fruits"] < whisker_min:
    #         fruits_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Fruits: " + str(row["Fruits"]))
    # for beer in fruits_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Hoppy
    # df2.boxplot(column=["Hoppy"], grid=False)
    # stats = boxplot_stats(df2["Hoppy"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Hoppy"] > whisker_max:
    #         hoppy_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Hoppy: " + str(row["Hoppy"]))
    #     if row["Hoppy"] < whisker_min:
    #         hoppy_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Hoppy: " + str(row["Hoppy"]))
    # for beer in hoppy_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Spices
    # df2.boxplot(column=["Spices"], grid=False)
    # stats = boxplot_stats(df2["Spices"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Spices"] > whisker_max:
    #         spices_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Spices: " + str(row["Spices"]))
    #     if row["Spices"] < whisker_min:
    #         spices_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Spices: " + str(row["Spices"]))
    # for beer in spices_outliers:
    #     print(beer)
    # plt.show()
    #
    # # Malty
    # df2.boxplot(column=["Malty"], grid=False)
    # stats = boxplot_stats(df2["Malty"].values)
    # whisker_max = stats[0]["whishi"]
    # whisker_min = stats[0]["whislo"]
    # for row in df.iloc:
    #     if row["Malty"] > whisker_max:
    #         malty_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Malty: " + str(row["Malty"]))
    #     if row["Malty"] < whisker_min:
    #         malty_outliers.append(
    #             "'" + row['Name'] + "'" + " from " + row['Brewery'] + " - Malty: " + str(row["Malty"]))
    # for beer in malty_outliers:
    #     print(beer)
    # plt.show()

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


    """
    Name: Beer's name
    Key: ID
    Style: Beer's style
    Style Key: a unique key assigned to each beer style
    Brewery: name of the beer's source
    Description: Notes on the beer (if available)
    ABV: Alcohol by Volume
    Astringency: refers to the puckery or drying sensation created in the mouth and throat
    Ave Rating: the average rating of the beer at the time of collection
    Min IBU: The minimum IBU value each beer can possess. IBU was not a value available for each beer, but the IBU range for each style was.
    Max IBU: The maximum IBU value each beer can possess. IBU was not a value available for each beer, but the IBU range for each style was.
    Body: refers to the weight or thickness of a beer
    Alcohol: I cannot find information about this category. I assume it measures the prevalence of the taste of alcohol
    Bitter: the bitterness level of the beer's taste
    Sweet: the sweetness level of the beer's taste
    Sour: the sourness level of the beer's taste
    Salty: the saltiness level of the beer's taste
    Fruits: the fruitiness level of the beer's taste
    Hoppy: the 'hoppy-ness' level of the beer's taste
    Spices: the spices level of the beer's taste
    Malty: the maltiness level of the beer's taste    
    """