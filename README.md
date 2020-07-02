# Analysis and Prediction for Airbnb Demand in Berlin

## Introduction
Airbnb gained a lot of popularity in the past decade and is a widely used platform for the booking of touristic acommodations in major cities. The evergrowing popularity for acommodations in Berlin, Germany's most populous city and a popular destination for tourists around the world, can be seen in the following graph.

![Sum of Reviews on Airbnb Acommodations in Berlin over Time](/images/vollständige_Zeitreihe.png)

It can be noted that bookings, approximated by the sum of reviews on acommodations, follow a clear upward trend (if we ignore COVID-19 pandemic's drastic impact on reviews in February and March 2020). Furthermore, a seasonal pattern is evident when observing the peaks at turns of years and low points around the Christmas period.

By using the number of reviews as an approximation for the demand of Airbnb accommodations, this analysis gives important insights into market developments.

## Data
The data for Berlin was gratefully retrieved from [Inside Airbnb](http://insideairbnb.com/get-the-data.html). For this analysis, the datasets on listings and reviews were utilized.

As already stated, demand for Airbnb acommodations is modelled through the number of written reviews. Users usually have 14 days for writing and submitting a review. According to the company, every second booking receives a review, thus making reviews a good indicator for actual bookings. As can be seen in the bar chart below, most reviews are written on a Sunday. Due to the uneven distribution weekly data is created for this analysis.

![Weekday of Submitted Reviews](/images/Verteilung_Reviews_auf_Tage.png)

In order to exclude recent developments caused by the COVID-19 pandemic which would bias the estimates, the data only includes observations up until February 2020. Data is split into a trainings and a testdataset. The ratio is 85:15, constituting a threshold in the early months of 2019. This can eminently be seen in the following plot which depicts the split.

![Weekday of Submitted Reviews](/images/Einteilung_Training_Test.png)

 
