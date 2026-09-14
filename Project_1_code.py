import numpy as np
import statsmodels.api as sm
import pandas as pd

'''Project 1 by Orsolya Bosáková'''

def WLSR(y, x, k, x_0):
    '''This is a weighted least squares regression function that takes in 
    the response variable y, the predictor variable x, the number of 
    neighbours k, and the point of interest x_0. It returns the predicted 
    value and standard error at x_0.'''
    D = [] 
    for i in range(len(x)):
        # finding the distance between x_0 and each x value
        dist = np.abs(x[i] - x_0)
        D.append((dist))
    
    # sorting it by distance and taking the k nearest neighbours
    neigh = sorted(D)[:k]
    d_max = max(neigh)

    for j in range(len(D)):
        # normalizing the distance by the maximum distance
        # and applying the tricube weight function
        D[j] = D[j] / d_max
        if D[j] >= 1:
            D[j] = 0
        else:
            D[j] = (1 - D[j] ** 3) ** 3
    
    # adding a constant to the predictor variable for the intercept
    x = sm.add_constant(x)

    # fitting the weighted least squares regression model
    # i hope it is okay to use the wls function
    model = sm.WLS(y, x, weights=D)
    results = model.fit()

    # predicting the value at x_0
    int_pred = [1, x_0]

    pred_info = results.get_prediction(int_pred)

    # separating the predicted value and standard error from the prediction results
    pred = pred_info.predicted_mean[0]
    se = pred_info.se_mean[0]

    return pred, se

#Mathematical note:
'''The sm.WLS or weighted least squares functions works as follows: Using our weights (matrix D), we can find the k nearest 
neighbours to our observation, which the model will be basing its prediction on. Then within the function the variance-covariance
matrix of beta is calculated as Var(beta) = sigma^2*(X^T@D@X)^-1, where X is a n by 2 matrix where the first column is 1 for
the intercept and the second column is the beta_1 values, sigma^2 is the weighted residual variance. Then the response at point 
x_0 is y_0 = x^T_0*beta, then we can calculate the var(y_0) by combining the formulas for var(y_0) and var_(beta), giving us the
formula that WLS uses to calculate the standard error.'''

#Programming note:
'''When calling the function pred_info = results.get_prediction(int_pred), it returns a multitude of values, like the
predicted mean of the expected value, the se of the mean prediction, the se of an individual observation, CI upper and 
lower bounds etc. We want to extract the se of the mean and the predicted mean of the expected value. These are both 
stored as an array, so to get a float variable, we specify that we want the 1st or 0th element saved for later.'''

# reading in the data
cleandata = pd.read_csv("Data/pollution_cleaneddata.csv")

# separating the predictor and response variables
x_val = cleandata['POOR'].values
y_val = cleandata['MORT'].values

percent = [10, 18, 25]

# using the function
get_results = [WLSR(y_val, x_val, k = 20, x_0 = p) for p in percent]

# separating the predicted values and standard errors
pred = [result[0] for result in get_results]
se = [result[1] for result in get_results]

# creating a dataframe to display the results
df_out = pd.DataFrame({"POOR (%)": percent, "Predicted MORT": pred, "SE": se})

print(df_out)

'''Results: 
   POOR (%)  Predicted MORT         SE
0        10      900.607323   8.091129
1        18      956.767778   6.747982
2        25     1010.030441  10.567187
'''


