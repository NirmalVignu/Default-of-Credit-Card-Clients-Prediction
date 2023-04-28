## End to End ML Project on Default of Credit Card Clients Prediction(Classification Problem)

### created an environment
```
conda create -p venv python==3.8

conda activate venv/
```
### Install all the neccessary libraries
```
pip install -r requirements.txt

```
### Probelm Statement

**This project is aimed at predicting the case of customers default payments in Taiwan. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. We can use the K-S chart to evaluate which customers will default on their credit card payments**

### Data Set Information:

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:

*  X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
*  X2: Gender (1 = male; 2 = female).
*  X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
*  X4: Marital status (1 = married; 2 = single; 3 = others).
*  X5: Age (year).
*  X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
*  X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
*  X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

### Model Report

```
Model Name: LogisticRegression
Accuracy:	0.8087777777777778
ROC AUC SCORE:	0.6005464401669759
Confusion Matrix:	[[6825 215] [1506 454]]
Precision:	0.6786248131539612
```

```
As our data set is imbalanced i.e,
 % of values are <=50K = 77.07182343065395
 % of values are >50K = 22.928176569346054

 tried giving class weight parameter and balancing using SMOTE both are not performing better than the normal approach so after careful observation from the different implementations logistic regression without balancing taken as best model and as it is imbalanced we have to look at the following statistics to determine model performance 
 ROC AUC SCORE, Confusion Matrix, Precision,Recall, F1 Score for which our model gives better results than the Base Line Score which i taken into COnsiderattion for evaluation and below is the comparision between the base line score and the acquired results
```
### Base Line Score 
```                    
Precision: 0.248
ROC AUC Score: 0.50
```
### Acquired Results
```
Precision:	0.7173713235294118
ROC AUC SCORE:	0.6005464401669759
```



