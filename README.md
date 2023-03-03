# Credit_Risk_Analysis With Supervised Machine Learning
## Overview
Credit risk is an unbalanced classification problem. aMy job is to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, you’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results
The Classification reports for each type of ML model, mainly on the Balanced Accuracy Score (the “geo” label) and the Harmonic Mean of the Precision and Sensitivity (the “f1” label)

### Random Oversampler Model
This model is quite precise with identifying low-risk loans, although not so well for high-risk loans which means a large number of false positives. The sensitivity (“rec” label) is much better, meaning a reduction in the number of false negatives. The Balanced Accuracy and F1 scores are not indicative of a good model.

### SMOTE Oversampler Model
This model shares many features of the Random Oversampler, but the Balanced Accuracy and F1 scores are slightly higher.

### Centroid Clusters Undersampler Model
Although the precision and sensitivity scores are similar to the oversampler models, this model’s Balanced Accuracy and F1 scores are significantly lower. This would be an intuitive result as lower sample sizes tend to reduce accuracy (as long as overfitting is not happening with the larger sample sizes).

### SMOTEENN Combination Model
This model, although using both oversampling and undersampling, has very similar results to both oversampling models.

### Balanced Random Forest Bootstrap Model
This model randomly undersamples each iteration of its decision tree, and was set to run 100 estimators. The precision of low-risk loans was slightly higher than the other models, but the sensitivity is markedly improved. This model is much less likely to generate false negatives for both high- and low-risk loans. The Balanced Accuracy and F1 scores are much improved as well.

### Easy Ensemble AdaBoost Model
This model uses different bootstrap samples and uses random undersampling for balance, and was also set to run 100 estimators. This model showed the highest amount of precision for high-risk loans, as well as the best Balanced Accuracy and F1 scores.

## Summary
Comparing all the models based on Balanced Accuracy and F1 scores should result in a clear choice of using the Easy Ensemble AdaBoost Model for a ML algorithm with this data set. The concern of overfitting data for this model is tempered by the fact that it uses random undersampling for each bootstrap.

I don’t recommend using any of these models, because of the high probability of false positives, these models may continue the decades-old lending practices that have shaped the socio economic landscape we are dealing with today. False positive labels for high risk loans have made a drastic impact on geographical, racial, and economic tensions.

More experimentation needs to be done using the previously mentioned feature importances and a wider variety of ML algorithms before any financial institution implements them.
