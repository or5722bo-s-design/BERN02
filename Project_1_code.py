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
    model = sm.WLS(y, x, weights=D)
    results = model.fit()

    # predicting the value at x_0
    int_pred = [1, x_0]

    pred_info = results.get_prediction(int_pred)

    # separating the predicted value and standard error from the prediction results
    pred = pred_info.predicted_mean[0]
    se = pred_info.se_mean[0]

    return pred, se

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

