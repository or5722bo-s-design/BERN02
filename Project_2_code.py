import numpy as np
import scipy as sc
import pandas as pd

'''Project 2 by Orsolya Bosáková'''

def nll_Poisson(params, x, y):
    '''The negative log likelihood function for poisson 
    distribution, that takes three arguments, params with is an array
    of two elements, x and y which are the independent and 
    dependent variables respectively, and returns the negative 
    log likelihood value.'''
    beta0, beta1 = params #unpacking the parameters

    #calculating the negative log likelihood value
    term_1 = y * (beta0 + beta1 * x)
    term_2 = np.exp(beta0 + beta1 * x)
    term_3 = np.log(sc.special.factorial(y))
    nll = np.sum(- term_1 + term_2 + term_3)

    return nll

#reading in the data and separating the independent and dependent variables
cleandata = pd.read_csv("Data/bird_count.csv")
years = cleandata["yr"].values
y_val = cleandata["count"].values
#adjusting the years to start from 0
x_val = years - years[0]

#initial guess for the parameters of the poisson distribution
params = [0.0, 0.0]

#fitting the poisson regression model using maximum likelihood estimation
max_likelihood = sc.optimize.minimize(nll_Poisson, params, args=(x_val, y_val))

#unpacking the estimated parameters from the optimization result
beta0_hat, beta1_hat = max_likelihood.x

#calculating the predicted values of the response variable
y_hat = np.exp(beta0_hat + beta1_hat * x_val)

#generating three random samples from the poisson distribution 
#with the predicted values as the mean
sample_1 = np.random.poisson(y_hat)
sample_2 = np.random.poisson(y_hat)
sample_3 = np.random.poisson(y_hat)

#saving the generated samples to a new csv file
df1 = pd.DataFrame({"yr": years, "sample": 1, "count": sample_1})
df2 = pd.DataFrame({"yr": years, "sample": 2, "count": sample_2})
df3 = pd.DataFrame({"yr": years, "sample": 3, "count": sample_3})

df = pd.concat([df1, df2, df3], ignore_index=True)

df.to_csv("bird_count_hw.csv", index=False)

