
# coding: utf-8

# # Linear Regression
# 
# *Linear Regression* is one of the simplest yet fundamental statistical learning techniques. It is a great initial step towards more advanced and computationally demanding methods. 
# 
# This article aims to cover a statistically sound approach to Linear Regression and its inferences while tying these to popular statistical packages and reproducing the results.
# 
# We first begin with a brief description of Linear Regression and move on to investigate it in light of a dataset.

# ## 1 - Description
# 
# Linear regression examines the relationaship between a dependent variable and one or more independent variables. Linear regression with $p$ independent variables focusses on fitting a straight line in $p+1$-dimensions that passes as close as possible to the data points in order to reduce error.
# 
# General Characteristics:
# 
# - A supervised learning technique
# - Useful for predicting a quantitative response
# - Linear Regression attempts to fit a function to predict a response variable
# - The problem is reduced to a parametric problem of finding a set of parameters
# - The function shape is limited (as a function of the parameters)

# ## 2- Advertising and Housing Datasets
# 
# Here we will use two datasets in order to get a feel of what Linear Regression is capable of.
# 
# First we use the Advertising dataset which is obtained from http://www-bcf.usc.edu/~gareth/ISL/data.html and contains 200 datapoints of sales of a particular product, and TV, newspaper and radio advertising budgets (all figures are in units of $1,000s). We will predict sales of a product given its advertising budgets.
# 
# Then we use the HousePrice dataset which is obtained from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data and contains 1460 houses along with many properties (only quantitative properties) including their sales prices. We will preduct the sale price of a property given certain parameters that characterise it.
# 
# First we import the required libraries

# In[5]:


# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from numpy.random import RandomState
import math

import statsmodels.api as sm
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import random


# Then we import the datasets

# In[6]:


# Import Advertising dataset (http://www-bcf.usc.edu/~gareth/ISL/data.html)
advert = pd.read_csv("Advertising.csv").iloc[:,1:]

# Import House Prices dataset - Only quantitative fields and cleaned (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
housePrice = pd.read_csv("HousePrice.csv").iloc[:,1:]


# In[7]:


print("Number of observations (n) in advertising file =",advert.shape[0])
print("Number of predictor variables (p) in advertising file =",advert.shape[1]-1)
print()
print("Advertising.csv")


# In[8]:


print("Number of observations (n) in house-prices file =",housePrice.shape[0])
print("Number of predictor variables (p) in house-prices file =",housePrice.shape[1]-1)
print()
print("HousePrice.csv")


# For the Advertising dataset the response variable is "sales". The predictor variables are "TV", "radio" and "newspaper". It's useful to visually inspect the data and see how each variable relates to the others. Using seaborn we can produce a pairplot of the data seen below:

# In[9]:


ax = sns.pairplot(data=advert)


# By looking at a pairplot to see the simple relationships between the variables, we see a strong positive correlation between sales and TV. A similar relationship between sales and radio is also observed. Newspaper and radio seem to have a slight positive correlation also. We can use the Pearson correlation given by:
# 
# $$corr=\frac{Cov(X,Y)}{\sigma_{X}\sigma_{Y}}$$
# 
# where $X$ and $Y$ are random variables, $Cov(X,Y)$ is the Covariance of $X$ and $Y$ and $\sigma_{X}$ is the standard deviation of $X$. This allows us to  examine the correlations between the parameters as seen in the correlation matrix below.

# In[10]:


advert.corr()


# We may want to fit a line to this data which is as close as possible. We describe the Linear Regression model next and then apply it to this data.

# ## 3- Linear Regression
# 
# The idea behind *Linear Regression* is that we reduce the problem of estimating the response variable, $Y$ = sales, by assuming there is a linear function of the predictor variables, $X_1$ = TV, $X_2$ = radio and $X_3$ = newspaper which describes $Y$. This reduces the problem to that of solving for the parameters $\beta_0$, $\beta_1$, $\beta_2$ and $\beta_3$ in the equation:
# 
# $$Y \approx \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \epsilon$$
# 
# where $\epsilon$ is an error term. After approximating the coefficients $\beta_i$ as $\hat{\beta}_i$, we obtain an approximation, $\hat{Y}$ of $Y$. The coefficients $\hat{\beta}_i$ are obtained using the observed realisations of the random variables $X_i$. Namely, $X_i = (x_{1i},x_{2i},x_{3i},...,x_{ni})$ are n observations of $X_i$ where $i = 1,2,...,p$. 
# 
# We first limit the problem to $p=1$. For example, we are looking to estimate the coefficients in the equation
# 
# $$Y \approx \beta_0 + \beta_1 X_1 + \epsilon$$
# 
# using the $n$ data points $(x_{11},y_{11}),(x_{21},y_{21}),...,(x_{n1},y_{n1})$. We can define the prediction discrepency of a particular prediction as the difference between the observed value and the predicted value. This is representated in mathematical notation for observation $i$ as $y_i - \hat{y}_i$. Letting $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1$ we have $y_i - \hat{y}_i = \epsilon_i$. i.e. the error in the prediction of point observation $i$ (also called the ith *residual*).
# 
# In summary, we are looking for a straight line to fit to the following data points as well as possible:

# In[11]:


# Get the figure handle and set figure size
fig = plt.figure(figsize=(8,8))

# Get the axis
axes = fig.add_axes([0.1,0.1,1,1])

# Plot onto the axis
axes.scatter(data=advert, x='TV', y='sales')

# Set the labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_1(x)')
axes.set_title('The relationship between Y = Sales and X = TV in the advertising dataset')
plt.show()


# In order to calculate appropriate values for parameters $\beta_i$, we would need a method of defining what it means for a line to be a good fit. A popular method is "Ordinary Least Squares". This method relies on minimising the Residual Sum of Squared errors (RSS). i.e. we are looking to minimise $RSS = \sum_{i=1}^n \epsilon_i^2$. While this intuitively makes sense, this can also be arrived at using a *Maximum Likelihood Estimation* (MLE) approach (see Appendix A2).
# 
# For the 1-parameter case we have that (the semi-colon below means 'the value of the parameters' given 'the data we have observed') 
# 
# $$RSS(\hat{\beta}_0,\hat{\beta}_1;X) = \sum_{i=1}^n \epsilon_i^2 = \sum_{i=1}^n (y_i-\hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$
# 
# We would like to find the parameters $(\beta_0,\beta_1)$ which minimise RSS. We first find the partial derivates:
# 
# $$\frac{\partial RSS}{\partial \hat{\beta_0}} = -2 [ \sum_{i=1}^n y_i - \sum_{i=1}^n \hat{\beta}_0 - \sum_{i=1}^n \hat{\beta}_1 x_i]$$
# 
# $$\frac{\partial RSS}{\partial \hat{\beta_1}} = -2 [ \sum_{i=1}^n y_i x_i - \sum_{i=1}^n \hat{\beta}_0 x_i - \sum_{i=1}^n \hat{\beta}_1 x_i^2]$$
# 
# Then setting these to zero and solving
# 
# $$\frac{\partial RSS}{\partial \hat{\beta_0}} = 0 \implies  \hat{\beta}_0 = \frac{\sum_{i=1}^n y_i - \hat{\beta}_1 \sum_{i=1}^n y_i}{n} = \frac{n \bar{y} - \hat{\beta}_1 n \bar{x}}{n} = \bar{y} - \hat{\beta}_1 \bar{x}$$
# 
# $$\frac{\partial RSS}{\partial \hat{\beta_1}} = 0 \implies  \sum_{i=1}^n y_i x_i - \hat{\beta}_0 \sum_{i=1}^n x_i - \hat{\beta}_1 \sum_{i=1}^n x_i^2 = 0$$
# 
# $$\implies \hat{\beta}_1 = \frac{n \bar{y} \bar{x} - \sum_{i=1}^n y_i x_i}{n \bar{x}^2 - \sum_{i=1}^n x_i^2} = \frac{\sum_{i=1}^n y_i x_i - n \bar{y} \bar{x}}{\sum_{i=1}^n x_i^2 - n \bar{x}^2} = \frac{\sum_{i=1}^n y_i x_i - n \bar{y} \bar{x} - n \bar{y} \bar{x} + n\bar{y} \bar{x}}{\sum_{i=1}^n x_i^2 - n \bar{x}^2 -n\bar{x}^2 + n\bar{x}^2}$$
# 
# $$= \frac{\sum_{i=1}^n x_i y_i - \sum_{i=1}^n y_i \bar{x} - \sum_{i=1}^n x_i \bar{y}  + \sum_{i=1}^n \bar{y} \bar{x}}{\sum_{i=1}^n x_i^2 - \sum_{i=1}^n x_i \bar{x} - \sum_{i=1}^n x_i \bar{x} + \sum_{i=1}^n \bar{x}^2},$$
# 
# where we used $n\bar{y} \bar{x} = \sum_{i=1}^n y_i \bar{x} = \sum_{i=1}^n x_i \bar{y}$ and $n\bar{x}^2 = n\bar{x} \bar{x} = \sum_{i=1}^n x_i \bar{x}$. Factorising
# 
# $$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$
# 
# Additionally, we can show that the point $(\bar{x},\bar{y})$ lies on the regression line (see Appendix A3).
# 
# We have now found the values of $(\hat{\beta}_0,\hat{\beta}_1)$ which corresponds to the extrema of RSS. We will still need to show that this is indeed a minima.
# 
# From Calculus, we know that if $\frac{\partial^2 RSS}{\partial \hat{\beta}_0 ^2} \frac{\partial^2 RSS}{\partial \hat{\beta}_1 ^2} - (\frac{\partial^2 RSS}{\partial \hat{\beta}_0 \partial \hat{\beta}_1})^2 > 0$, this is an extrema and not an inflexion point. Additionally, if $\frac{\partial^2 RSS}{\partial \hat{\beta}_0 ^2} > 0$ and $\frac{\partial^2 RSS}{\partial \hat{\beta}_1 ^2} > 0$ this is a minima.
# 
# We have that
# 
# $$\frac{\partial^2 RSS}{\partial \hat{\beta}_0 ^2} = 2n > 0$$
# $$\frac{\partial^2 RSS}{\partial \hat{\beta}_1 ^2} = 2 \sum_{i=1}^n x_i^2 > 0$$
# $$\frac{\partial^2 RSS}{\partial \hat{\beta}_0 \partial \hat{\beta}_1} = 2 \sum_{i=1}^n x_i$$
# 
# So,
# 
# $\frac{\partial^2 RSS}{\partial \hat{\beta}_0 ^2} \frac{\partial^2 RSS}{\partial \hat{\beta}_1 ^2} - (\frac{\partial^2 RSS}{\partial \hat{\beta}_0 \partial \hat{\beta}_1})^2 = (2n) (2 \sum_{i=1}^n x_i^2) - (2 \sum_{i=1}^n x_i)^2 > 0 \; \forall \; n>1$ (see Appendix A1).
# 
# This means that this is indeed a minima (since we have satisfied the conditions stated above). 
# 
# The equation
# 
# $$\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1$$
# 
# then defines a straight line of best fit which minimises the expected value of the errors (residuals). From the form of this line, we can see that $\hat{\beta}_0$ corresponds to the value of $\hat{Y}$ if the independent variable $X_1$ is zero. $\hat{\beta}_1$ is then the gradient.
# 
# In the following we construct 3 functions dependent on a single independent variable and attach an error term and calculate the best fit. The three functions are chosen as:
# 
# 1- $f_1(x) = 4.67 + 5.07*x$
# 
# 2- $f_2(x) = 4.67 + 5.07*x^2$
# 
# 3- $f_3(x) = 4.67 + 5.07*sin(x/20)$

# In[12]:


#f_1(x)=4.67+5.07∗x
def f_1(x):
    return 4.67 + 5.07*x

#f_2(x)=4.67+5.07∗x2
def f_2(x):
    return 4.67 + 5.07*x**2

#f_3(x)=4.67+5.07∗sin(x/20)
def f_3(x):
    return 4.67 + 5.07*math.sin(x/20)


# In[13]:


# Set the seed
r = np.random.RandomState(101)

# Choose 1000 random observations for x between 0 and 100
X = 100*r.rand(1000)

#Error term with sigma = 10, mu = 0, randn samples from the standard normal distribution
E_1 = 10*r.randn(1000)

#Error term with sigma = 500, mu = 0
E_2 = 500*r.randn(1000)

#Error term with sigma = 1, mu = 0
E_3 = 1*r.randn(1000)

#Response variables
Y_1 = list(map(f_1,X))+E_1
Y_2 = list(map(f_2,X))+E_2
Y_3 = list(map(f_3,X))+E_3


# In the above, *s $\times$ r.randn(n)* samples n points from the $N(0,s^2)$ distribution. First we look at what $f_1$ looks like

# In[14]:


# Plot
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_1,'.')

# Set labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_1(x)')
axes.set_title('Scatter plot of f_1')

plt.show()


# The task is to fit the model $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1$ to the data. We know that 
# 
# $$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$
# 
# and
# 
# $$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$
# 
# We can calculate these as below

# In[15]:


#Find the mean of the data for f_1
x_bar1 = np.mean(X)
y_bar1 = np.mean(Y_1)

numerator = 0
denominator = 0

for i in range(len(Y_1)):
    # Add to the numerator for beta_1
    numerator += (X[i] - x_bar1)*(Y_1[i] - y_bar1)
    
    # Add to the denominator for beta_1
    denominator += (X[i] - x_bar1)**2
    
beta1_1 = numerator/denominator
beta1_0 = y_bar1 - beta1_1*x_bar1

print('Y = {beta_0} + {beta_1} * X'.      format(beta_0 = beta1_0, beta_1 = beta1_1))


# Below, we see how the line defined by the equation above fits the data for $f_1$

# In[16]:


# 1000 linearly spaced numbers
x1 = np.linspace(0,99,1000) 

# The equation using the betas above
y1 = beta1_0 + beta1_1 * x1 

# Plot the observed data
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_1,'.')

# Plot the regression line
axes.plot(x1,y1)

# Set labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_1(x)')
axes.set_title('A plot of the data for f_1 and the regression line')

plt.show()


# Let's see what the residuals look like by plotting them. The residuals require the knowledge of the actual response variables so that we can compare them with the predicted response variables. So we use the regression line above to predict the response variable using the observed predictor variables. Then we plot them using a histogram to gain some insight into their distribution

# In[17]:


# The fitted values are the predicted values given the observed values
y1_fitted = beta1_0 + beta1_1 * X

# The residuals are the differences between our predicted values and 
# the observed responses
Res_1 = y1_fitted - Y_1

# Plot the residuals
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.hist(Res_1)

# Set labels and title
axes.set_xlabel('Residuals')
axes.set_ylabel('Count')
axes.set_title('A histogram of the residuals for f_1 when fitted with the regression line')

plt.show()

print('This is roughly a normal distribution with mean {mean} \nand standard deviation {std}'.format(mean=np.mean(Res_1),std=np.std(Res_1)))


# Since the residuals are roughly normally distributed, our model may be a good choice. In fact, the standard deviation for the residuals was roughly equal to the standard deviation for the error term when we constructed the function $f_1$. A model may suffer from two types of error: 
# * error due to a discrepancy between the chosen function shape (here a linear model) and the true function shape (this is the reducible error), and 
# * error due to random noise (this is the irreducible error). We can see here that the residuals are from irreducible error. 
# 
# Above we fitted a linear model to our 'designed' linear data. The error terms we expect to get are irreducible and a result of the error term E1 added above.
# 
# Now let's do the same for f_2.

# In[18]:


# Get figure handle
fig = plt.figure(figsize=(8,8))

# Get axis handle and specify size
axes = fig.add_axes([0.1,0.1,1,1])

# Plot onto this axis
axes.plot(X,Y_2,'.')

# Set the axis labels
axes.set_xlabel('x')
axes.set_ylabel('f_2(x)')
axes.set_title('Scatter plot of f_2')


# In[19]:


#Find the mean of the data for f_2
x_bar2 = np.mean(X)
y_bar2 = np.mean(Y_2)

numerator = 0
denominator = 0

for i in range(len(Y_2)):
    # Add to the numerator for beta_1
    numerator += (X[i] - x_bar2)*(Y_2[i] - y_bar2)
    
    # Add to the denominator for beta_1
    denominator += (X[i] - x_bar2)**2
    
beta2_1 = numerator/denominator
beta2_0 = y_bar2 - beta2_1*x_bar2

print('Y = {beta_0} + {beta_1} * X'.format(beta_0 = beta2_0, beta_1 = beta2_1))


# Below, we see how the line defined by the equation above fits the data for $f_2$

# In[20]:


# 1000 linearly spaced numbers
x2 = np.linspace(0,99,1000) 

# The predicted responses of these 1000 numbers
y2 = beta2_0 + beta2_1 * x2

# Plot
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_2,'.')
axes.plot(x2,y2)

# Set labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_2(x)')
axes.set_title('A plot of the data and the regression fit for f_2')

plt.show()


# We can then look at the residuals plot as we did before

# In[21]:


# The fitted values are the predicted values given the observed values
y2_fitted = beta2_0 + beta2_1 * X

# The residuals are the differences between our predicted values and 
# the observed responses
Res_2 = y2_fitted - Y_2


# Plot the residuals
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.hist(Res_2)

# Set labels and title
axes.set_xlabel('Residuals')
axes.set_ylabel('Count')
axes.set_title('A histogram of the residuals for f_2 when fitted with the regression line')

plt.show()

print('The residuals are certainly not from a normal distribution')


# This shows that the linear model we have chosen may not be a good choice. We can try $X^2$ as a parameter instead of $X$ in our linear model. This way, we are transforming an existing parameter to form a new parameter.

# In[22]:


# Create X^2 parameter
X_2 = X**2

#Find the mean of the data for f_2
x_bar22 = np.mean(X_2)
y_bar22 = np.mean(Y_2)

numerator = 0
denominator = 0

for i in range(len(Y_2)):
    # Calculate the numerator for beta_1
    numerator += (X_2[i] - x_bar22)*(Y_2[i] - y_bar22)
    
    # Calculate the denominator for beta_1
    denominator += (X_2[i] - x_bar22)**2
    
beta22_1 = numerator/denominator
beta22_0 = y_bar22 - beta22_1*x_bar22

print('Y = {beta_0} + {beta_1} * X^2'.format(beta_0 = beta22_0, beta_1 = beta22_1))


# Below, we see how the new line defined by the equation above fits the data for $f_2$

# In[23]:


# 1000 linearly spaced numbers
x22 = np.linspace(0,99,1000)

# Predicted responses to the 1000 numbers
y22 = beta22_0 + beta22_1 * ((x22)**2)

# Plot this regression line and the data
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_2,'.')
axes.plot(x22,y22)

# Set labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_2(x)')
axes.set_title('A plot of the data and the new regression fit for f_2')
plt.show()


# We see a much better fit. Now we investigate the residuals to see if the new regression fit using $X^2$ as a parameter yields residuals that look more normally distributed as has been assumed by the model architecture

# In[24]:


# The fitted values are the predicted values given the observed values
y22_fitted = beta22_0 + beta22_1 * X**2

# The residuals are the differences between our predicted values and 
# the observed responses
Res_22 = y22_fitted - Y_2

# Plot the residuals
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.hist(Res_22)

# Set labels and title
axes.set_xlabel('Residuals')
axes.set_ylabel('Count')
axes.set_title('A histogram of the residuals for f_2 when fitted with the new regression line')

plt.show()

print('This is roughly a normal distribution with mean {mean} and standard deviation {std}'      .format(mean=np.mean(Res_22),std=np.std(Res_22)))


# This shows that we can transform an independent variable and apply linear regression in order to *regress* the response variable onto the transformed explanatory variable. This increases the power of linear regression techniques. Note also that the standard deviation from the residual distribution is close to the 500 for the errors when the function was created.
# 
# Now let's apply linear regression to f_3 in a similar manner

# In[25]:


# Get figure handle
fig = plt.figure(figsize=(8,8))

# Get axis handle and specify size
axes = fig.add_axes([0.1,0.1,1,1])

# Plot onto this axis
axes.plot(X,Y_3,'.')

# Set the axis labels
axes.set_xlabel('x')
axes.set_ylabel('f_3(x)')
axes.set_title('Scatter plot of f_3')

plt.show()


# It is very clear from the above scatter plot that we will not be able to get away with fitting a linear line to the data. This is a hint that we should use transformed variables. But let's carry out a linear fit to show that the results can be misleading when we only consider the residuals plot to assess the quality of fit

# In[26]:


#Find the mean of the data for f_3
x_bar3 = np.mean(X)
y_bar3 = np.mean(Y_3)

numerator = 0
denominator = 0

for i in range(len(Y_3)):
    numerator += (X[i] - x_bar3)*(Y_3[i] - y_bar3)
    denominator += (X[i] - x_bar3)**2
    
beta3_1 = numerator/denominator
beta3_0 = y_bar3 - beta3_1*x_bar3

print('Y = {beta_0} + {beta_1} * X'.format(beta_0 = beta3_0, beta_1 = beta3_1))


# Below, we see how the line defined by the equation above fits the data for $f_3$

# In[27]:


# 1000 linearly spaced numbers
x3 = np.linspace(0,99,1000) 

# Predict the response for those numbers
y3 = beta3_0 + beta3_1 * x3

# Plot both the data and the fit
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_3,'.')
axes.plot(x3,y3)

# Set the labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_3(x)')
axes.set_title('A plot of the data and the new regression fit for f_3')

plt.show()


# We now assess the residuals

# In[28]:


# The fitted values are the predicted values given the observed values
y3_fitted = beta3_0 + beta3_1 * X

# The residuals are the differences between our predicted values and 
# the observed responses
Res_3 = y3_fitted - Y_3

# Plot the residuals
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.hist(Res_3)

# Set labels and title
axes.set_xlabel('Residuals')
axes.set_ylabel('Count')
axes.set_title('A histogram of the residuals for f_3 when fitted with the regression line')

plt.show()

print('This not a normal distribution but it is not that far off.')


# ### Alternative View
# 
# Often in Machine Learning, we pose a hypothesis ($h_{\theta}(X)$) and a cost function ($J(\theta)$) and proceed to minimise this cost function. Here, $X$ is the data and $\theta$ is a vector of parameters (such as the $\beta$ in the Linear Regression models above).
# 
# For Linear Regression as stated above, the hypothesis function is that there is a straight line passing through all the data points:
# 
# $$h_{\theta}(X) = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + \theta_3 X_3 + ... = X \theta$$
# 
# The Cost function is the least squares sum residuals (eventually written in index notation):
# 
# $$J(\theta) = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (h_{\theta}(X^{(i)}) - Y^{(i)})^2 = (X \theta - Y)^T (X \theta - Y) = (X \theta)^T X\theta - 2 (X \theta)^T Y + Y^T Y = \theta_j x_{ji} x_{ij} \theta_{j} - 2 \theta_j x_{ji} y_i$$
# 
# where the superscript $^{(i)}$ refers to the ith observation. Taking the derivative of the cost function:
# 
# $$\frac{\partial J(\theta)}{\partial \theta_k} = 2 x_{ki} x_{ik} \theta_k - 2 x_{ki} y_i$$
# 
# Setting this to zero for all $k$ and solving:
# 
# $$\theta = (X^T X)^{-1} X^T Y$$
# 
# Let's test this out with the sine curve above by considering $X$,$X^2$ and $X^3$ as predictor variables.

# In[29]:


# Calculate x^2
X2 = X**2

# Calculate X^3
X3 = X**3

# Combine into a single array (n X 3)
X_full = np.concatenate((X.reshape(-1,1),X2.reshape(-1,1),X3.reshape(-1,1)),axis=1)

# Create transpose (3 X n)
X_fullT = X_full.transpose()

# Calculate X^T X
XTX = X_fullT.dot(X_full)

# calculate inverse of XTX
XTX_inv = np.linalg.inv(XTX)

# Calculate theta
theta = XTX_inv.dot(X_fullT.dot(Y_3))

print('Y_3 = {} * X + {} * X^2 + {} * X^3'.format(theta[0],theta[1],theta[2]))


# We can add a constant to the above by create an extra predictor full of ones

# In[30]:


# Calculate x^2
X2 = X**2

# Calculate X^3
X3 = X**3

# Combine into a single array (n X 4)
X_full = np.concatenate((np.ones((Y_3.shape[0],1)),X.reshape(-1,1),X2.reshape(-1,1),X3.reshape(-1,1)),axis=1)

# Create transpose (4 X n)
X_fullT = X_full.transpose()

# Calculate X^T X (4 X 4)
XTX = X_fullT.dot(X_full)

# calculate inverse of XTX (4 X 4)
XTX_inv = np.linalg.inv(XTX)

# Calculate theta
theta = XTX_inv.dot(X_fullT.dot(Y_3))

print('Y_3 = {} + {} * X + {} * X^2 + {} * X^3'.format(theta[0],theta[1],theta[2],theta[3]))


# We plot the original data along with this solution to the parameters to see how well it fits the data.

# In[31]:


# 1000 linearly spaced numbers
x3 = np.linspace(0,99,1000) 
x3_2 = x3**2
x3_3 = x3**3

# Predict the response for those numbers
y3 = theta[0] + theta[1] * x3 + theta[2] * x3_2 + theta[3] * x3_3

# Plot both the data and the fit
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_3,'.')
axes.plot(x3,y3)

# Set the labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_3(x)')
axes.set_title('A plot of the data and the new regression fit for f_3')

plt.show()


# #### $R^2$-Statistic
# 
# Even though a plot of the residuals above does not show a clear divergence from a normal distribution, it is clear from the predicted-observed plot that this is not a good model and does not fit the data in a satisfactory manner. We therefore need additional tools in order to asses the level of fit.
# 
# A metric we can use in order to assess the goodness of the fit is the *R-Squared* ($R^2$) statistic. The $R^2$ statistic measures the percentage of variability of the response variable that is explained by the explanatory variable. This is mathematically expressed as:
# 
# $$R^2 = \frac{TSS-RSS}{TSS}$$
# 
# where $TSS = \sum_{i=1}^n(y_i - \bar{y})^2$ is the *total sum of squares* and $RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2$ is the *residual sum of squares*.
# 
# Note: Another way to assess the lack of fit is through the *Residual Squared Error* $RSE=\sqrt{ \frac{ RSS }{ n-2 } }$. 
# 
# $R^2$, as the form above suggests, is the proportion of variance that is explained. For a simple linear regression with 1 parameter (see Appendix A4):
# 
# $$R^2 = Cor(X,Y)^2 = \left( \frac{ \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2 } } \right) ^2$$
# 
# However, for multiple linear regression this does not hold. It is not clear how to adapt the Correlation in order to explain the fit of a multiple regression model. $R^2$ however, is a clearly defined metric which is easily extended to multiple regression.
# 
# Below, we calculate this metric for $f_3$

# In[32]:


# TSS
TSS_3 = 0

# RSS
RSS_3 = 0

for i in range(len(X)):
    TSS_3 += (Y_3[i] - y_bar3)**2
    RSS_3 += (Y_3[i] - y3_fitted[i])**2

# R^2 for f_3
R_sq_3 = (TSS_3 - RSS_3)/TSS_3
print('R^2 = {}'.format(R_sq_3))


# This means that roughly 59% of the variability in $Y_3$ is explained by $X$. Let's calculate the $R^2$ statistic for all the models above. To do this, we create a function that accepts observed and fitted values and returns the TSS and RSS of the fit

# In[33]:


def TSS_RSS(y_observed,y_fitted):
    '''
    A function that calculates the TSS and RSS of a fit given observed 
        and fitted values
    y_observed := Observed data as a list
    y_fitted := Fitted data as a list
    output := A (TSS,RSS) tuple of floats
    '''
    
    # TSS
    TSS = 0
    
    # RSS
    RSS = 0
    
    # Get the mean of the observed values
    y_bar = np.mean(y_observed)

    for i in range(len(y_observed)):
        TSS += (y_observed[i] - y_bar)**2
        RSS += (y_observed[i] - y_fitted[i])**2
        
    return TSS,RSS


# Then we apply this function to the three fitted models

# In[34]:


# Calculate the TSS and RSS for the fitted regression line to f_1
TSS_1, RSS_1 = TSS_RSS(Y_1,y1_fitted)

# Calculate the R^2 for the fit to f_1
R_sq_1 = (TSS_1 - RSS_1)/TSS_1
print('Model for Y_1: Explanatory variable X for Y_1 - R^2 = {}'      .format(R_sq_1))


# Calculate the TSS and RSS for the fitted regression line to f_2
TSS_2,RSS_2 = TSS_RSS(Y_2,y2_fitted)

# Calculate the R^2 for the fit to f_2
R_sq_2 = (TSS_2 - RSS_2)/TSS_2
print('Model for Y_2: Explanatory variable X for Y_2 - R^2 = {}'      .format(R_sq_2))


# Calculate the TSS and RSS for the new fitted regression line to f_2
TSS_22,RSS_22 = TSS_RSS(Y_2,y22_fitted)
   
# Calculate the R^2 for the new fit to f_2
R_sq_22 = (TSS_22 - RSS_22)/TSS_22
print('Model for Y_2: Explanatory variable X^2 for Y_2 - R^2 = {}'      .format(R_sq_22))


# Calculate the TSS and RSS for the fitted regression line to f_3
TSS_3,RSS_3 = TSS_RSS(Y_3,y3_fitted)

# Calculate the R^2 for the fit to f_3
R_sq_3 = (TSS_3 - RSS_3)/TSS_3
print('Model for Y_3: Explanatory variable X for Y_3 - R^2 = {}'      .format(R_sq_3))


# From the above we can see that the model for $Y_1$ that is linear in $X$ is satisfactory; The model for $Y_2$ that is non-linear explains more variability of the response variable than the linear model (note that in this case, the $R^2$ metric alone wouldn't tell us whether the fit linear in $X$ was terrible. But along with the residual plot we would arrive at the correct conclusion); The model for $Y_3$ shows that we are probably not fitting the correct form of the function, i.e. we have introduced bias in that the real function is not of the form $a+bX$ for constants $a$ and $b$ and that applying a model non-linear in $X$ may provide a boost to the explained variance. We can try combinations of $X$, $X^2$, $X^3$ as well. We do this after we have introduced a much simpler way of obtaining the above fits using Scikit-Learn packages.
# 
# Below, we use *sklearn.linear_model.LinearRegression()* in order to fit and *sklearn.metrics.r2_score()* in order to calculate the $R^2$ statistic. We will see that the results match the manual results above

# In[35]:


# Create the model object
lm1 = LinearRegression()

# Fit this model to the data for f_1
lm1.fit(X.reshape(-1,1),Y_1.reshape(-1,1))

print('Model for Y_1: Explanatory variable X for Y_1')
print('beta_0 = {}'.format(lm1.intercept_[0]))
print('beta_1 = {}'.format(lm1.coef_[0][0]))

# Get the fitted values and print it
y1_fitted_sklearn = lm1.intercept_[0] + lm1.coef_[0][0]*X
print('R^2 = {}'.format(r2_score(Y_1,y1_fitted_sklearn)))

print()
print()

lm2 = LinearRegression()
lm2.fit(X.reshape(-1,1),Y_2.reshape(-1,1))
print('Model for Y_2: Explanatory variable X for Y_2')
print('beta_0 = {}'.format(lm2.intercept_[0]))
print('beta_1 = {}'.format(lm2.coef_[0][0]))
y2_fitted_sklearn = lm2.intercept_[0] + lm2.coef_[0][0]*X
print('R^2 = {}'.format(r2_score(Y_2,y2_fitted_sklearn)))

print()
print()

lm22 = LinearRegression()
lm22.fit((X**2).reshape(-1,1),Y_2.reshape(-1,1))
print('Model for Y_2: Explanatory variable X^2 for Y_2')
print('beta_0 = {}'.format(lm22.intercept_[0]))
print('beta_1 = {}'.format(lm22.coef_[0][0]))
y22_fitted_sklearn = lm22.intercept_[0] + lm22.coef_[0][0]*X**2
print('R^2 = {}'.format(r2_score(Y_2,y22_fitted_sklearn)))

print()
print()

lm3 = LinearRegression()
lm3.fit(X.reshape(-1,1),Y_3.reshape(-1,1))
print('Model for Y_3: Explanatory variable X for Y_3')
print('beta_0 = {}'.format(lm3.intercept_[0]))
print('beta_1 = {}'.format(lm3.coef_[0][0]))
y3_fitted_sklearn = lm3.intercept_[0] + lm3.coef_[0][0]*X
print('R^2 = {}'.format(r2_score(Y_3,y3_fitted_sklearn)))

print()
print()

# Now we try adding the variables X,X^2 and X^3

#Create transformed variables
X2 = X**2
X3 = X**3

lm32 = LinearRegression()
X3_collection = pd.concat([pd.DataFrame(X,columns=['X']),                pd.DataFrame(X**2,columns=['X2']),                pd.DataFrame(X**3,columns=['X3'])],axis=1)
lm32.fit(X3_collection,Y_3.reshape(-1,1))
print('Model for Y_3: Explanatory variables X,X^2,X^3 for Y_3')
print('beta_0 = {}'.format(lm32.intercept_[0]))
print('beta_1 = {}'.format(lm32.coef_[0][0]))
print('beta_2 = {}'.format(lm32.coef_[0][1]))
print('beta_3 = {}'.format(lm32.coef_[0][2]))
y32_fitted_sklearn = lm32.intercept_[0] + lm32.coef_[0][0]*X +                     lm32.coef_[0][1]*X**2 + lm32.coef_[0][2]*X**3
print('R^2 = {}'.format(r2_score(Y_3,y32_fitted_sklearn)))


# In the above, we fit a model using 3 explanatory variables, namely $X$, $X^2$, $X^3$ with coefficients $\beta_1$, $\beta_2$, $\beta_3$ respectively. We can see that we have a much improved $R^2$ statistic for the fitted model to $f_3$ meaning we have managed to explain much more of the data using the transformed variables we have created. We can plot the model to see how well it follows the response variable.

# In[36]:


# 1000 linearly spaced numbers
x32 = np.linspace(0,99,1000) 
y32 = lm32.intercept_[0] + lm32.coef_[0][0]*x32 + lm32.coef_[0][1]*x32**2    + lm32.coef_[0][2]*x32**3

# Plot the data and the fit
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X,Y_3,'.')
axes.plot(x32,y32)

# Set the lables and title
axes.set_xlabel('x')
axes.set_ylabel('f_3(x)')
axes.set_title('A plot of the data and the model using X,X^2 and X^3 as predictors')

plt.show()


# We can also check the residuals plot

# In[37]:


# Calculate the fitted values using the observed values
y32_fitted_sklearn = lm32.intercept_[0] + lm32.coef_[0][0]*X +                     lm32.coef_[0][1]*X**2 + lm32.coef_[0][2]*X**3

# Calculate the residuals
Res_32 = y32_fitted_sklearn - Y_3

# Plot the residuals
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.hist(Res_32)

# Set the lables and title
axes.set_xlabel('Residuals')
axes.set_ylabel('Count')
axes.set_title('A plot of the residuals of the model using X,X^2 and X^3 as predictors')

plt.show()

print('This is roughly a normal distribution with mean {mean} and standard deviation {std}'.format(mean=np.mean(Res_32),std=np.std(Res_32)))


# It is not a surprise that we were able to fit a function of the form $f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$. Using taylor expansion, $f(x) = sin(x)$ estimated around the point $x=0$ is
# 
# $$f(x=0) = f(0) + f^{(1)}(0)x + f^{(2)}(0)x^2/(2!) + f^{(3)}(0)x^3/(3!) + O(x^4)$$
# $$= \sin(0) + \cos(0)x - \sin(0)x^2/(2!) -\cos(0)x^3/(3!)$$
# $$= x - x^3/(6)$$
# 
# If we apply Taylor series expansion to $f(x) = 4.67 + 5.07 sin(x/20)$ instead:
# 
# $$f(x=0) = 4.67 + \frac{5.07}{20}\cos(0)x-\frac{5.07}{20^3}\cos(0)x^3/(3!)=4.67 + 0.25x - 1 \times 10^{-4} x^3$$
# 
# Let's plot this along with the above for smaller values of X for which this approximation of sin(x) is acceptable.

# In[38]:


# 1000 linearly spaced numbers
x32 = np.linspace(0,50,1000) 

# Predictions
y32 = lm32.intercept_[0] + lm32.coef_[0][0]*x32 + lm32.coef_[0][1]*x32**2    + lm32.coef_[0][2]*x32**3

# Prediction using Taylor expansion
y_taylor_32 = 4.67 + (5.07/20)*x32 + 0*x32**2 - (5.07/(20**3 * 6))*x32**3

# Only get the observed predictors and response where the predictors are less 
# than 50
X_small = list(filter(lambda x: x < 50,X))
Y_small = Y_3[list(map(lambda x: x < 50,X))]

# Plot the data, the fitted model and the taylor expansion
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(X_small,Y_small,'.',label='Data')
axes.plot(x32,y32,label='Linear Model')
axes.plot(x32,y_taylor_32,label='Taylor Expansion')

# Set the labels and title
axes.set_xlabel('x')
axes.set_ylabel('f_3(x)')
axes.set_title('A comparison of the fits from the regression model and the Taylor exansion of f_3')

# Add the legend
axes.legend()

plt.show()


# #### Statistical significance of regression coefficients
# In addition to the $R^2$ statistic, it is useful to assess whether a variable is statistically significant. To do this for a variable $X$ with coefficient $\beta_1$, we test the null hypothesis
# 
# $$H_O: \beta_1 = 0$$
# 
# against
# 
# $$H_A: \beta_1 \neq 0$$
# 
# For the first model we have the fitted model

# In[39]:


print('f(x) = {} + {} X'.format(lm1.intercept_[0],lm1.coef_[0][0]))


# The standard errors of the estimators $\hat{\beta}_0$ and $\hat{\beta}_1$ for the coefficients have the form (See Appendix A5):
# 
# $$SE(\beta_0) = \sqrt{\sigma^2 \left[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right]} \approx  RSE\sqrt{ \left[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right]}$$
# 
# where RSE is the *residual standard error* estimating the population $\sigma =\sqrt{Var(\epsilon)}$ and has the form $RSE = \sqrt{\frac{\sum_{i=1}^n \epsilon_i^2}{n-2}} = \sqrt{\frac{RSS}{n-2}}$.
# 
# 
# In addition we can show that:
# 
# $$SE(\beta_1) = \sqrt{   \frac{ \sigma^2 }{ \sum_{i=1}^n (x_i - \bar{x})^2 }   } \approx RSE\sqrt{   \frac{ 1 }{ \sum_{i=1}^n (x_i - \bar{x})^2 }   }$$
# 
# 
# Using the standard errors, we can then conduct the hypothesis test above as a t-test. We have that 
# 
# $$\frac{ \hat{\beta_0} - \beta_0^{(0)} }{ SE(\beta_0) } \sim t_{n-2}$$
# 
# $$\frac{ \hat{\beta_1} - \beta_1^{(0)} }{ SE(\beta_1) } \sim t_{n-2}$$
# 
# where $^{(0)}$ denotes the null value (the null hypothesis above sets both $\beta_0^{(0)}=0$ and $\beta_1^{(0)}=0$).

# In[40]:


# number of observations n
n = len(X)

# residual standard error
RSE_1 = np.sqrt(RSS_1/(n-2))

# variance of x = sum (x_i - x_bar)^2. Note that this is the 
# population variance calculation
# so we would need to multiply by n
varx_1 = np.var(X)

# mean of x
meanx_1 = np.mean(X)

SE_beta_0 = RSE_1 * np.sqrt(1.0/n + meanx_1**2/(n*varx_1))
SE_beta_1 = RSE_1 * np.sqrt(1.0/(n*varx_1))

print('SE(beta_0) = {}, SE(beta_1) = {}'.format(SE_beta_0,SE_beta_1))

# null hypothesis
betanull_0 = 0
betanull_1 = 0

tstatistic1_0 = (beta1_0 - betanull_0)/SE_beta_0
tstatistic1_1 = (beta1_1 - betanull_1)/SE_beta_1

print('beta_0 t-statistic = {}'.format(tstatistic1_0))
print('beta_1 t-statistic = {}'.format(tstatistic1_1))

# p-value
# the following function calculates the area under the student t pdf with 
# 2 degrees of freedom that is less than -4.303
stats.t.cdf(-4.303,2)

# calculate the p-value using the tstatistic and degrees of freedom n-2
pval1_0 = stats.t.cdf(-tstatistic1_0,n-2)
pval1_1 = stats.t.cdf(-tstatistic1_1,n-2)

print('p-value for beta_0 = {}'.format(pval1_0))
print('p-value for beta_1 = {}'.format(pval1_1))
print('These are both statistically significant!')


# We can put this into a function

# In[41]:


def calcpvalue(X,y_observed,y_fitted,beta_0,beta_1,betanull_0,betanull_1):
    '''
    A function to calculate whether the coefficients in a model with 1 
        variable is statistically significant.
    X = a list for the data for the variable
    y_observed = the observed values for the response variable
    y_fitted = the predicted values of the model
    beta_0 = the intercept of the model
    beta_1 = the coefficient of the explanatory variable in the model
    betanull_0 = null hypothesis value for the intercept (usually 0)
    betanull_1 = null hypothesis value for the coefficient of the response 
        variable (usually 0)
    '''
    # number of observations n
    n = len(X)

    # calculate RSS
    temp,RSS = TSS_RSS(y_observed,y_fitted)
    
    # residual standard error
    RSE = np.sqrt(RSS/(n-2))

    # variance of x = sum (x_i - x_bar)^2. Note that this is the population 
    # variance calculation
    # so we would need to multiply by n
    varx = np.var(X)

    # mean of x
    meanx = np.mean(X)

    SE_beta_0 = RSE * np.sqrt(1.0/n + meanx**2/(n*varx))
    SE_beta_1 = RSE * np.sqrt(1.0/(n*varx))

    print('SE(beta_0) = {}, SE(beta_1) = {}'.format(SE_beta_0,SE_beta_1))

    # null hypothesis
    betanull_0 = 0
    betanull_1 = 0

    tstatistic1_0 = (beta_0 - betanull_0)/SE_beta_0
    tstatistic1_1 = (beta_1 - betanull_1)/SE_beta_1

    print('beta_0 t-statistic = {}'.format(tstatistic1_0))
    print('beta_1 t-statistic = {}'.format(tstatistic1_1))

    # p-value

    # calculate the p-value using the tstatistic and degrees of freedom n-2
    # Multiply by 2 since it's a 2 tailed test
    if(tstatistic1_0 > 0):
        pval_0 = stats.t.cdf(-tstatistic1_0,n-2)*2
    else:
        pval_0 = stats.t.cdf(tstatistic1_0,n-2)*2
        
    if(tstatistic1_1 > 0):
        pval_1 = stats.t.cdf(-tstatistic1_1,n-2)*2
    else:
        pval_1 = stats.t.cdf(tstatistic1_1,n-2)*2

    print('p-value for beta_0 = {}'.format(pval_0))
    print('p-value for beta_1 = {}'.format(pval_1))
    if((pval_0 <= 0.05) and (pval_1 <=0.05)):
        print('These are both statistically significant!')
    elif(pval_0 <= 0.05):
        print('Only beta_0 is statistically significant!')
    elif(pval_1 <= 0.05):
        print('Only beta_1 is statistically significant!')
    else:
        print('The parameters of this model are not statistically significant!')


# We can do the same calculations for significance for all the models using this function

# In[42]:


print('Model for Y_1: Explanatory variable X for Y_1')
calcpvalue(X,Y_1,y1_fitted,beta1_0,beta1_1,0,0)

print()
print()

print('Model for Y_2: Explanatory variable X for Y_2')
calcpvalue(X,Y_2,y2_fitted,beta2_0,beta2_1,0,0)

print()
print()

print('Model for Y_2: Explanatory variable X^2 for Y_2')
calcpvalue(X**2,Y_2,y22_fitted,beta22_0,beta22_1,0,0)

print()
print()

print('Model for Y_3: Explanatory variable X for Y_3')
calcpvalue(X,Y_3,y3_fitted,beta3_0,beta3_1,0,0)


# We can use the statsmodels.api to verify our results

# In[43]:


print('Model for Y_1: Explanatory variable X for Y_1')

# add a column of ones to X
X_new = sm.add_constant(X)

# ordinary least squares approach to optimisation
est = sm.OLS(Y_1, X_new)

# fit the data to the model using OLS
est2 = est.fit()

# print a summary of the model
print(est2.summary())

print()
print()

#re-run the above for all the models

print('Model for Y_2: Explanatory variable X for Y_2')
X_new = sm.add_constant(X)
est = sm.OLS(Y_2, X_new)
est2 = est.fit()
print(est2.summary())

print()
print()

print('Model for Y_2: Explanatory variable X^2 for Y_2')
X_new = sm.add_constant(X**2)
est = sm.OLS(Y_2, X_new)
est2 = est.fit()
print(est2.summary())

print()
print()

print('Model for Y_3: Explanatory variable X for Y_3')
X_new = sm.add_constant(X)
est = sm.OLS(Y_3, X_new)
est2 = est.fit()
print(est2.summary())

print()
print()

print('Model for Y_3: Explanatory variables X,X^2,X^3 for Y_3')
# concatenate multiple variables
X_new = sm.add_constant(pd.concat([pd.DataFrame(X,columns=['X']),                                   pd.DataFrame(X**2,columns=['X2']),                                   pd.DataFrame(X**3,columns=['X3'])],axis=1))
est = sm.OLS(Y_3, X_new)
est2 = est.fit()
print(est2.summary())


# It looks like the intercept for *Model for Y_2: Explanatory variable X^2 for Y_2* is not statistically significant. The intercept can then be omitted from the model and fitted again.

# In[44]:


print('Model for Y_2: Explanatory variable X^2 for Y_2')
est = sm.OLS(Y_2, X**2)
est2 = est.fit()
print(est2.summary())


# This is a good fit also

# In[45]:


x23 = np.linspace(0,99,1000) # 1000 linearly spaced numbers
y23 = est2.params[0] * x23**2

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(X,Y_2,'.')
axes.set_xlabel('x')
axes.set_ylabel('f_2(x)')
axes.plot(x23,y23)


# If we set $\beta_0=0$ in the derivation for $\hat{\beta_0}$ and $\hat{\beta_1}$ earlier in the article, we would have obtained the equation
# 
# $$\hat{\beta_1} = \frac{\sum_{i=1}^n y_i x_i}{\sum_{i=1}^n x_i^2}$$
# 
# Using this equation, we can reproduce the statsmodels solution above. Note that removing $\beta_0$ has changed $\beta_1$ slightly:

# In[46]:


# remember that we are fitting the variable X^2
sum1 = np.sum(Y_2*X**2)
sum2 = np.sum(X**4)

beta23_1 = sum1/sum2

print('Y ~ {} X^2'.format(beta23_1))


# #### F-Statistic
# 
# The F-Statistic answers the question 'Is there evidence that at least one of the explanatory variables is related to the response variable?'. This corresponds to a hypothesis test with:
# 
# $$H_O: \beta_0, \beta_1, ..., \beta_p = 0$$
# $$H_A: \text{at least one of } \beta_i \text{ is non-zero}$$
# 
# The F-Statistic has the form:
# 
# $$F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)}$$
# 
# where p is the number of explanatory variables/parameters. 
# 
# If $H_O$ is not true, the numerator in the above equation becomes larger, i.e. F > 1. If $H_0$ is true, then the F-Statistic is close to 1.
# 
# (PROOF of this - take expectation of numerator and denominator and these are both equal to Var($\epsilon$). If $H_A$ is true then the numerator > Var($\epsilon$))
# 
# We can use this to calculate the F-Statistics of the above models:

# In[47]:


def FStat(n,p,TSS,RSS):
    F = ((TSS-RSS)/p)/(RSS/(n-p-1))
    print('The F-Statistic is {}'.format(F))


# In[48]:


# we didn't calculate the last model ourselves, we used sklearn 
# so we retrieve the coefficients
beta32_0 = lm32.intercept_[0]
beta32_1 = lm32.coef_[0][0]
beta32_2 = lm32.coef_[0][1]
beta32_3 = lm32.coef_[0][2]


# In[49]:


print('Model for Y_1: Explanatory variable X for Y_1')
FStat(len(X),1,TSS_1,RSS_1)

print()
print()

#re-run the above for all the models

print('Model for Y_2: Explanatory variable X for Y_2')
FStat(len(X),1,TSS_2,RSS_2)

print()
print()

print('Model for Y_2: Explanatory variable X^2 for Y_2')
FStat(len(X),1,TSS_22,RSS_22)

print()
print()

print('Model for Y_3: Explanatory variable X for Y_3')
FStat(len(X),1,TSS_3,RSS_3)

print()
print()

TSS_32,RSS_32 = TSS_RSS(Y_3,y32_fitted_sklearn)

print('Model for Y_3: Explanatory variables X,X^2,X^3 for Y_3')
# now we have 3 explanatory variables
FStat(len(X),3,TSS_32,RSS_32)


# These match the *statsmodels* outputs. We can also find the p-value of a coefficient/intercept using the F-Statistic. The F-Statistic formula becomes:
# 
# $$F = \frac{(RSS_0 - RSS)/q}{RSS/(n-p-1)}$$
# 
# where $RSS_0$ is the residual sum of squares for the model with $q$ removed parameters. The corresponding hypothesis test is then
# 
# $$H_0: \{ \beta_i = 0 \} \text{ where i takes on the q removed parameters}$$
# $$H_A: \text{at least one of those q parameters is non-zero}$$
# 
# Above, we ran a model for Y_2 which had an intercept, coefficient of X^2 and RSS of:

# In[50]:


beta22_0, beta22_1, RSS_22


# Here, we are going to calculate the p-value of the intercept for Y_2 when we try to fit an intercept as well as $X^2$. We do this by first fitting the full model including the intercept and getting the RSS value, then we fit the model without the intercept and get the RSS value. The Coefficient of X^2 and RSS for the model without the intercept was calculated to be

# In[51]:


TSS_23,RSS_23 = TSS_RSS(Y_2,beta23_1 * X**2)
print('beta_1 = {}, RSS_0 = {}'.format(beta23_1,RSS_23))


# We now create a function to apply the formula shown above for calculating the F-Statistic for comparing models

# In[52]:


def FStatCompare(n,p,q,RSS0,RSS):
    '''
    A function to calculate the F-Statistic when we are comparing models 
        with different number of parameters.
    RSS0 is a sub-model of RSS
    '''
    F = ((RSS0-RSS)/q)/(RSS/(n-p-1))
    print('The F-Statistic is {}'.format(F))
    return F


# Now we can confirm the p-value for the intercept

# In[53]:


# This is the fitted values for the model with no intercept
Y23_fitted = beta23_1 * X**2

# These are the TSS and RSS for this model with no intercept
TSS_2_test,RSS_2_test = TSS_RSS(Y_2,Y23_fitted)

# RSS_22 is the RSS for the model with the intercept. RSS_23 is the RSS
# for the model without the intercept. We have p = 0 and q = 1 (i.e. we have 
# removed 1 parameter but there was only 1 parameter to begin with)
F = FStatCompare(len(X),0,1,RSS_23,RSS_22)

# the following function calculates the area underneath the cdf F-distribution 
# with dfn(degrees of freedom in the numerator)=1, 
# dfd(degrees of freedom in the denominator)=len(X)-2 less than 0.5
stats.f.cdf(0.5,1,len(X)-2)

print('The p-value of the intercept is {}'.format(1-stats.f.cdf(F,1,len(X)-2)))


# Note that above, we removed the intercept and used the F-Statistic to calculate the p-value for the intercept. We can also remove the coefficient of X^2 and calculate the p-value of this coefficient using the same procedure as above. First fit the model as we have done before

# In[54]:


lmOnlyIntercept = LinearRegression()
lmOnlyIntercept.fit((X*0).reshape(-1,1),Y_2.reshape(-1,1))
print('Model for Y_2: No explanatory variable for Y_2')
print('beta_0 = {}'.format(lmOnlyIntercept.intercept_[0]))
yOnlyIntercept_fitted_sklearn = lmOnlyIntercept.intercept_[0] + X*0
print('R^2 = {}'.format(r2_score(Y_2,yOnlyIntercept_fitted_sklearn)))


# Next, calculate the RSS for this model we have just fitted

# In[55]:


TSS_OnlyIntercept,RSS_OnlyIntercept = TSS_RSS(Y_2,yOnlyIntercept_fitted_sklearn)
print('beta_0 = {}, RSS_0 = {}'.format(lmOnlyIntercept.intercept_[0],                                       RSS_OnlyIntercept))


# And now we calculate the p-value of the coefficient of X^2

# In[56]:


# These are the TSS and RSS for this model with only intercept
TSS_2_test,RSS_2_test = TSS_RSS(Y_2,yOnlyIntercept_fitted_sklearn)

# RSS_22 is the RSS for the model with the intercept. RSS_23 is the RSS
# for the model without the intercept. We have p = 0 and q = 1 (i.e. we have 
# removed 1 parameter but there was only 1 parameter to begin with)
F = FStatCompare(len(X),0,1,RSS_2_test,RSS_22)

# the following function calculates the area underneath the cdf F-distribution 
# with dfn(degrees of freedom in the numerator)=1, 
# dfd(degrees of freedom in the denominator)=len(X)-2 less than 0.5
stats.f.cdf(0.5,1,len(X)-2)

print('The p-value of the X^2 coefficient is {}'      .format(1-stats.f.cdf(F,1,len(X)-2)))


# ### Synergy Effect
# 
# Suppose we have the following function
# 
# $$f(x)=4.67+2*X_1+3*X_2+5.07∗X_1*X_2$$
# 
# We can see that there is a mixed term '$X_1 X_2$'. This is called a synergy effect.
# 
# Let's define this function and plot it

# In[57]:


# We will need to plot in 3D
from mpl_toolkits.mplot3d import Axes3D

#f(x)=4.67+2*X_1+30*X_2+5.07∗X_1*X_2
def f(x1,x2):
    return 4.67+2*x1+30*x2+5.07*x1*x2
# Set the seed
r = np.random.RandomState(101)
X_1 = 100*r.rand(1000)
X_2 = -20*r.rand(1000)

#Error term with sigma = 10, mu = 0
E = 100*r.randn(1000)

#Response variables
Y = list(map(f,X_1,X_2))+E

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(X_2,Y,'.')
axes.set_xlabel('x')
axes.set_ylabel('f(x)')


# In[58]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_1, X_2, Y, c='r', marker='o')

ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.set_zlabel('Y')

plt.tight_layout()


# Suppose we continued to fit a linear regression model with parameters $X_1$ and $X_2$ with the assumption that there is no synergy effect.

# In[59]:


print('Model for Y: Explanatory variable X_1 and X_2 for Y')
X_new = sm.add_constant(pd.concat([pd.DataFrame(X_1,columns=['X_1']),                                   pd.DataFrame(X_2,columns=['X_2'])],axis=1))
est = sm.OLS(Y, X_new)
est2 = est.fit()
print(est2.summary())


# The above output shows that the $R^2$ is almost 87% with both $X_1$ and $X_2$ being statistically significant. Below, we show that including the synergy term $X_1 X_2$ into the model as well greatly improves the $R^2$ metric.

# In[60]:


print('Model for f: Explanatory variables X_1, X_2 and X_1 * X_2 for Y_2')
X_new = sm.add_constant(pd.concat([pd.DataFrame(X_1,columns=['X_1']),                                   pd.DataFrame(X_2,columns=['X_2']),                                   pd.DataFrame(X_1*X_2,columns=['X_12'])],axis=1))
est = sm.OLS(Y, X_new)
est2 = est.fit()
print(est2.summary())


# We have seen above that adding a term $X_1 X_2$ significantly increased the $R^2$ statistic. Instead of adding this joint term, what would be the effect on $R^2$ if we added random noise? We see the effect below.

# In[61]:


# Set the seed
r = np.random.RandomState(11)

#Error term with sigma = 10, mu = 0
noise = 100*r.randn(1000)

print('Model for Y: Explanatory variable X_1 and X_2 for Y')
X_new = sm.add_constant(pd.concat([pd.DataFrame(X_1,columns=['X_1']),                                   pd.DataFrame(X_2,columns=['X_2']),                                  pd.DataFrame(noise,columns=['Noise'])],axis=1))
est = sm.OLS(Y, X_new)
est2 = est.fit()
print(est2.summary())


# Including an unrelated, random noise term to the model increases the $R^2$ statistic. This makes sense since when fitting the model to the training data, in the worst case, the model could choose a predictor's coefficient to be zero. This means that the $R^2$ statistic for the training data should never decrease as a function of the number of predictors. The main reason for introducing such a metric is to gauge how well the model describes the population from which our data originates from. However, if it never decreases then how can it be determined whether the added parameter is useful or not? 
# 
# In order to cater for this, the Adjusted $R^2$ metric can be used. This metric applies a penalty to the usual $R^2$ the more predictors that are used. This way, it is not possible that the Adjusted $R^2$ can increase indefinitely. At some point, the contribution to the $R^2$ of adding a new predictor will be overcome by the penalty attributed to adding that new parameter. The Adjusted $R^2$ is as follows:
# 
# $$ \text{Adjusted }R^2 = 1 - \frac{ RSS/(n - p - 1) }{ TSS/(n - 1) } $$
# 
# where $p$ is the number of predictors. It can be seen in the above model that the Adjusted $R^2$ did not increase with the addition of another predictor.
# 
# Another approach we can apply to take into account that the test $R^2$ will always be smaller than the training $R^2$, is to divide the data we have into a training set and a testing set. We can then train the model on the training set and test it on the unseen testing set in order to determine how well it has performed.
# 
# We tackle this in the next section.

# ### Cross Validation
# 
# 

# *Cross Validation* is a technique to estimate how well a model will perform on unseen data. As mentioned in the previous section, the entire data set available can be divided into two: a training set and a testing set. The question then becomes, 'what portion of the dataset should be the training set?'. This question can be expressed as follows:
# 
# - Let the number of observations be $n$, then the trainnig set is $n-k$ where $k \in [1,n-1]$
# 
# The reason this question is important is that the choice of $k$ greatly influences the bias in our cross validation. If $k = \lfloor n/2 \rfloor$ then the test error will be greatly overestimated since the final model will be trained on $n$ observations, not $\lfloor n/2 \rfloor$ observations. On the other hand, if $k=1$, the variance of our test error will be very large since the technique will depend greatly on which observation we chose as the test observation. 
# 
# Going further, we can divide the entire dataset into roughly $n/k$ subsets. We can then run $n/k$ different cross validations leaving a different subset as the test set at each iteration. The test error (or $R^2$) can then be approximated as the average of the different subset test errors. This immediately means that if $n$ is large, choosing to assess the model performance using cross validation with $k=1$ could be computationally intense. Therefore, a value for $k$ somewhere in the range $(1,\lfloor n/2 \rfloor)$ may be wiser.
# 
# To start things off, let's fit a linear regression model to the house prices dataset and test it on a portion of the data.

# In[63]:


housePrice.columns


# We use train test split (using 33% of the dataset as a test set) to calculate the MSE on the test set

# In[64]:


# The predictor and response
X = housePrice['YearBuilt'].values.reshape(-1,1)
y = housePrice['SalePrice'].values.reshape(-1,1)

# Make 33% of this dataset a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Linear Regression model object
model = LinearRegression()

# Fit this model using the training data
model.fit(X_train,y_train)

# Predict
predictions = model.predict(X_test)

# Get the RSS
tss,rss = TSS_RSS(y_test,predictions)

# The MSE is RSS/n_test
MSE = rss/len(y_test)

print('The MSE is {}'.format(MSE))

# Plot the predictions
#plt.scatter(X_test,y_test)
#plt.scatter(X_test,predictions)


# Let's see what the MSE is when we use the 'LotArea' predictor to predict 'SalePrice'.

# In[65]:


# The predictor and response
X = housePrice['LotArea'].values.reshape(-1,1)
y = housePrice['SalePrice'].values.reshape(-1,1)

# Make 33% of this dataset a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Linear Regression model object
model = LinearRegression()

# Fit this model using the training data
model.fit(X_train,y_train)

# Predict
predictions = model.predict(X_test)

# Get the RSS
tss,rss = TSS_RSS(y_test,predictions)

# The MSE is RSS/n_test
MSE = rss/len(y_test)

print('The MSE is {}'.format(MSE))

# Plot the predictions
#plt.scatter(X_test,y_test)
#plt.scatter(X_test,predictions)


# An important point to note in the above MSE calculations is that these MSE results are highly biased. We used a train - test split of 33%. However, in reality, we have the full dataset to train our model on. This means that the above is overestimating the test MSE of the model. In other words, by using only a subset of our dataset to train our model, we are not making use of the full power of the data we have. We can go to the other extreme and select one single observation from our data set of n observations as a test set and the remaining n-1 observations as a training set. This is called Leave One Out Cross Validation (LOOCV). We do that below.

# In[66]:


# The predictor and response
X = housePrice['LotArea']
y = housePrice['SalePrice']

# Select a random element to be the test set
r = random.SystemRandom()
testint = r.randint(0,len(X))

# The train set
X_train = X.copy().values.reshape(-1,1)
y_train = y.copy().values.reshape(-1,1)

# The test set is that one observation
X_test = X_train[testint]
X_test = X_test.reshape(1,-1)

# The train set is all observations except that one observation
np.delete(X_train,testint)

# The test set consist of one response
y_test = y_train[testint]
np.delete(y_train,testint)

# The Linear Regression model object
model = LinearRegression()

# Fit the model
model.fit(X_train,y_train)

# Predict
predictions = model.predict(X_test)

# Get the MSE. MSE = RSS/n
tss,rss = TSS_RSS(y_test,predictions)
MSE = rss/len(y)
print('MSE = {}'.format(MSE))

# Plot
#plt.scatter(X_test,y_test)
#plt.scatter(X_test,predictions)


# Each time we run the above code, we get a completely different test MSE. This is because the test MSE depends on which observation we chose to test the model on. So this reduces the bias to a minimum but has a large variance. We can iterate over all the cases where for each iteration we leave a different observation as a test observation. Then we calculate the average MSE over all these Cross Validations.

# First for 'YearBuilt' as a predictor variable then the 'LotArea' as a predictor variable.

# In[67]:


# The predictor and response
X = housePrice['YearBuilt']
y = housePrice['SalePrice']

# The MSE array. Each element is the MSE of a particular Cross Validation
MSE = []

# Perform LOOCV on the data using Linear Regression
for i in range(len(X)):
    # The training set
    X_train = X.copy().values.reshape(-1,1)
    y_train = y.copy().values.reshape(-1,1)
    
    # The test set is a single observation
    X_test = X_train[i]
    X_test = X_test.reshape(1,-1)
    X_train = np.delete(X_train,i)

    # The test set is a single observation
    y_test = y_train[i]
    y_test = y_test.reshape(1,-1)
    y_train = np.delete(y_train,i)

    # Train the model
    model = LinearRegression()

    # Fit
    model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))

    # Predict
    predictions = model.predict(X_test)

    # Calculate the MSE. MSE = RSS/n_test
    tss,rss = TSS_RSS(y_test,predictions)
    MSE.append(rss[0])

# Print the mean MSE value
print(np.mean(MSE))


# In[68]:


X = housePrice['LotArea']
y = housePrice['SalePrice']

MSE = []

for i in range(len(X)):
    X_train = X.copy().values.reshape(-1,1)
    y_train = y.copy().values.reshape(-1,1)
    
    X_test = X_train[i]
    X_test = X_test.reshape(1,-1)
    X_train = np.delete(X_train,i)

    y_test = y_train[i]
    y_test = y_test.reshape(1,-1)
    y_train = np.delete(y_train,i)

    # Train
    model = LinearRegression()

    # Fit
    model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))

    # Predict
    predictions = model.predict(X_test)

    # MSE
    tss,rss = TSS_RSS(y_test,predictions)
    MSE.append(rss[0])

print(np.mean(MSE))


# We can leverage the cross_val_score method to do the above cross validation for us

# In[70]:


X = housePrice['LotArea']
y = housePrice['SalePrice']

linregCVScores = cross_val_score(LinearRegression(),X.values.reshape(-1,1),y.values.reshape(-1,1),scoring='neg_mean_squared_error',cv=len(X))
-linregCVScores.mean()


# Let's observe now which approach (value of k in k-fold cross validation) predicts the test MSE best. We split our train and test data. Then estimate the test MSE using the training data.

# In[71]:


datasetMSEEstimatek_10 = []
datasetMSEEstimatek_20 = []
datasetMSEEstimatek_100 = []
datasetMSEActual = []

# The predictor and response
X = housePrice['LotArea'].values.reshape(-1,1)
y = housePrice['SalePrice'].values.reshape(-1,1)


for j in range(500):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=j)

    linregCVScores = cross_val_score(LinearRegression(),X_train.reshape(-1,1),y_train.reshape(-1,1),scoring='neg_mean_squared_error',cv=10)
    datasetMSEEstimatek_10.append(-linregCVScores.mean())
    
    linregCVScores = cross_val_score(LinearRegression(),X_train.reshape(-1,1),y_train.reshape(-1,1),scoring='neg_mean_squared_error',cv=20)
    datasetMSEEstimatek_20.append(-linregCVScores.mean())
    
    linregCVScores = cross_val_score(LinearRegression(),X_train.reshape(-1,1),y_train.reshape(-1,1),scoring='neg_mean_squared_error',cv=100)
    datasetMSEEstimatek_100.append(-linregCVScores.mean())
    
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    datasetMSEActual.append(mean_squared_error(y_test,predictions))
    
    if j%50 == 0:
        print('Step = {}'.format(j))
    
print('The mean MSE Estimation using K-fold CV with k = 10 is : {}'.format(np.mean(datasetMSEEstimatek_10)))
print('The mean MSE Estimation using K-fold CV with k = 20 is : {}'.format(np.mean(datasetMSEEstimatek_20)))
print('The mean MSE Estimation using K-fold CV with k = 100 is : {}'.format(np.mean(datasetMSEEstimatek_100)))
print('The actual MSE on this data set is : {}'.format(np.mean(datasetMSEActual)))

fig,axes = plt.subplots(nrows = 2,ncols = 2,sharex=True)
fig.set_size_inches(20,10)

axes[0][0].hist(list(map(math.log,datasetMSEEstimatek_10)))
axes[0][0].set_title('CV with k = 10')

axes[0][1].hist(list(map(math.log,datasetMSEEstimatek_20)))
axes[0][1].set_title('CV with k = 20')

axes[1][0].hist(list(map(math.log,datasetMSEEstimatek_100)))
axes[1][0].set_title('CV with k = 100')

axes[1][1].hist(list(map(math.log,datasetMSEActual)))
axes[1][1].set_title('Averaged Actual MSE over different train/test splits')


# In[72]:


datasetMSEEstimatek_10 = []
datasetMSEEstimatek_20 = []
datasetMSEEstimatek_100 = []
datasetMSEActual = []

# The predictor and response
X = housePrice['YearBuilt'].values.reshape(-1,1)
y = housePrice['SalePrice'].values.reshape(-1,1)


for j in range(500):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=j)

    linregCVScores = cross_val_score(LinearRegression(),X_train.reshape(-1,1),y_train.reshape(-1,1),scoring='neg_mean_squared_error',cv=10)
    datasetMSEEstimatek_10.append(-linregCVScores.mean())
    
    linregCVScores = cross_val_score(LinearRegression(),X_train.reshape(-1,1),y_train.reshape(-1,1),scoring='neg_mean_squared_error',cv=20)
    datasetMSEEstimatek_20.append(-linregCVScores.mean())
    
    linregCVScores = cross_val_score(LinearRegression(),X_train.reshape(-1,1),y_train.reshape(-1,1),scoring='neg_mean_squared_error',cv=100)
    datasetMSEEstimatek_100.append(-linregCVScores.mean())
    
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    datasetMSEActual.append(mean_squared_error(y_test,predictions))
    
    if j%50 == 0:
        print('Step = {}'.format(j))
    
print('The mean MSE Estimation using K-fold CV with k = 10 is : {}'.format(np.mean(datasetMSEEstimatek_10)))
print('The mean MSE Estimation using K-fold CV with k = 20 is : {}'.format(np.mean(datasetMSEEstimatek_20)))
print('The mean MSE Estimation using K-fold CV with k = 100 is : {}'.format(np.mean(datasetMSEEstimatek_100)))
print('The actual MSE on this data set is : {}'.format(np.mean(datasetMSEActual)))

fig,axes = plt.subplots(nrows = 2,ncols = 2,sharex=True)
fig.set_size_inches(20,10)

axes[0][0].hist(list(map(math.log,datasetMSEEstimatek_10)))
axes[0][0].set_title('CV with k = 10')

axes[0][1].hist(list(map(math.log,datasetMSEEstimatek_20)))
axes[0][1].set_title('CV with k = 20')

axes[1][0].hist(list(map(math.log,datasetMSEEstimatek_100)))
axes[1][0].set_title('CV with k = 100')

axes[1][1].hist(list(map(math.log,datasetMSEActual)))
axes[1][1].set_title('Averaged Actual MSE over different train/test splits')


# We can see above that the MSE when we choose the 'LotArea' predictor is not as good as using 'YearBuilt'. So in this case we choose 'YearBuilt' over 'LotArea' to include in our linear regression model. 
# 
# It can be seen that there is a general pattern in the above when comparing the MSE estimates from varying blocks (k) in Cross Validation. Namely, the larger k is, the more blocks we use to split the data up and the less portion of the data there is for the test set and the more iteration that is required per Cross Validation. For example, suppose n = 1000, when k = 10 we have 10 blocks with 100 observations per block. So we fit the model 10 times, each time leaving out a different k block when training. When comparing this with the case where k = 100, we have 100 blocks each with 10 observations. This means that the training set for each of these model fits is a lot closer to reality in that we will be using the entire dataset to fit the model. However, this comes at a computational cost, in this case we would need to run a model to fit and predict 100 times instead of 10. 
# 
# Depending on the computational cost of the model used, we may choose a smaller value of k = 10 when comparing the same model but with different parameters. 
# 
# We can try out all the predictors and choose the one that minimises the mean squared errors. This can be done by using a loop as below.

# In[73]:


# Run through each model in the correct order and run CV on it and save the best CV score
bestMeanCV = -1
bestMeanCVModel = []

X = housePrice.drop('SalePrice',axis=1)

# y is the response variable
y = housePrice['SalePrice']

for i in X.columns:
    # First set X to be the full set of remaining parameters
    X = housePrice.loc[:,i]
    
    linregCVScores = cross_val_score(LinearRegression(),X.values.reshape(-1,1),y,scoring='neg_mean_squared_error',cv=10)
    
    if bestMeanCV > -linregCVScores.mean():
        bestMeanCV = -linregCVScores.mean()
        bestMeanCVModel = i
    elif bestMeanCV == -1:
        bestMeanCV = -linregCVScores.mean()
        bestMeanCVModel = i
        
print('The final best model is {} and its TEST MSE is {}'.format(bestMeanCVModel,bestMeanCV))


# We can then iterate through the predictors adding it to the model each time in order to improve the test MSE of the model. For instance, in the above, we have selected as the first predictor in our model, the predictor 'OverallQual'. Next, we cycle through all the remaining predictors to include in our model along with 'OverallQual' and repeat. The final result will be a list of all predictors in the order they were added. Once we get to a point where adding another predictor to the model does not improve the test MSE, then we stop there.

# In[74]:


# Run through each model in the correct order and run CV on it and save the best CV score
bestMeanCV = -1
bestMeanCVModel = []
oldArraySize = 0

X = housePrice.drop('SalePrice',axis=1)

columnsArray = X.columns

# y is the response variable
y = housePrice['SalePrice']

while oldArraySize != len(X):
    bestPredictor = ''
    oldArraySize = len(X.columns)
    for i in columnsArray:
        thisModel = bestMeanCVModel.copy()
        thisModel.append(i)
        # First set X to be the full set of remaining parameters
        x = X.loc[:,thisModel]

        if len(x.columns) == 1:
            linregCVScores = cross_val_score(LinearRegression(),x.values.reshape(-1,1),y,scoring='neg_mean_squared_error',cv=10)
        else:
            linregCVScores = cross_val_score(LinearRegression(),x,y,scoring='neg_mean_squared_error',cv=10)
            
        if bestMeanCV > -linregCVScores.mean():
            bestMeanCV = -linregCVScores.mean()
            bestPredictor = i
        elif bestMeanCV == -1:
            bestMeanCV = -linregCVScores.mean()
            bestPredictor = i
    
    if bestPredictor not in columnsArray:
        break
        
    columnsArray.drop(bestPredictor)
    bestMeanCVModel.append(bestPredictor)
    print('{} was added with test MSE {}'.format(bestMeanCVModel[-1],bestMeanCV))

        
print('The final best model is {} and its TEST MSE is {}'.format(bestMeanCVModel,bestMeanCV))


# Our final model is now contained in bestMeanCVModel.

# ### Ridge Regression
# 
# Ridge Regression adds a twist to Linear Regression with the aim of reducing the variance of the model and managing multicollinearity. 
# 
# We begin with the normal equation as we did for Linear Regression and arrive at a method of calculating the parameters of the regression formula.

# As before, we pose a hypothesis ($h_{\theta}(X)$) and a cost function ($J(\theta)$) and proceed to minimise this cost function. Here, $X$ is the data and $\theta$ is a vector of parameters (such as the $\beta$ in the Linear Regression models above).
# 
# For Linear Regression as stated above, the hypothesis function is that there is a straight line passing through all the data points:
# 
# $$h_{\theta}(X) = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + \theta_3 X_3 + ... = X \theta$$
# 
# The Cost function is the least squares sum residuals (eventually written in index notation):
# 
# $$J(\theta) = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (h_{\theta}(X^{(i)}) - Y^{(i)})^2 = (X \theta - Y)^T (X \theta - Y) = (X \theta)^T X\theta - 2 (X \theta)^T Y + Y^T Y = \theta_j x_{ji} x_{ij} \theta_{j} - 2 \theta_j x_{ji} y_i$$
# 
# where the superscript $^{(i)}$ refers to the ith observation. The extra step we will be taking here is to add an additional term in this equation which is the L2 norm $\lambda||\theta||^2 = \lambda \sum_{i=1}^n \theta_i = \lambda \theta'^T \theta' = \lambda \theta'_j \theta'_j$, where $\theta'$ is the parameter vector $\theta$ but with the first term corresponding to the coefficient of the constant term set to zero and $\lambda$ is a scaling factor or shrinkage factor:
# 
# $$J(\theta) = \theta_j x_{ji} x_{ij} \theta_{j} - 2 \theta_j x_{ji} y_i + \lambda \theta'_j \theta'_j$$
# 
# Taking the derivative of the cost function:
# 
# $$\frac{\partial J(\theta)}{\partial \theta_k} = 2 x_{ki} x_{ik} \theta_k - 2 x_{ki} y_i + 2\lambda \theta'_k$$
# 
# where $\theta'_k = 0$ for $k=0$.
# 
# Setting this to zero for all $k$ and solving:
# 
# $$\theta = (X^T X + \lambda I')^{-1} X^T Y$$
# 
# where we have used the fact that $\theta' I = \theta I'$ with $I$ the identity matrix and $I'$ the identity matrix where element $I_{11} = 0$ (i.e. we are transfering effect of the first element of $\theta$ being zero to the identity matrix).
# 
# In summary, this additional regularisation term serves to attach a penalty to large coefficients in the minimisation process. It should be added that if we set $\lambda = 0$, no additional constraint is performed and we just get the Linear Regression solution.
# 
# Let's test this out the sklearn module:

# In[75]:


L = 0.2
normalize = False

# X is the predictor variable 
X = housePrice.drop('SalePrice',axis=1)[['LotArea','YearBuilt']]

# y is the response variable
y = housePrice['SalePrice']

# Convert the predictor dataframe and the response dataframe to arrays to be consistent with data type
X = np.concatenate((np.ones((X.shape[0],1)),np.array(X)),axis=1)
y = np.array(y)

ncols = X.shape[1]

if normalize:
    # standardise X if required
    for i in range(1,ncols):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])

    # standardise y if required
    y = (y - np.mean(y))/np.std(y)

# Create transpose (3 X n)
X_T = X.transpose()

# Calculate X^T X (3 X 3)
XTX = X_T.dot(X)

# Create I'
Id = np.eye(XTX.shape[0])
Id[0,0] = 0

# Add the shrinkage factor part
XTX = XTX + L*Id

# calculate inverse of XTX + lambda I' (3 X 3)
XTX_inv = np.linalg.inv(XTX)

# Calculate theta
theta = XTX_inv.dot(X_T.dot(y))

print('Y_3 = {} + {} * LotArea + {} * YearBuilt'.format(theta[0],theta[1],theta[2]))


# We can run this using the package and we see the exact same results

# In[77]:


# X is the predictor variable 
X = housePrice.drop('SalePrice',axis=1)[['LotArea','YearBuilt']]

# y is the response variable
y = housePrice['SalePrice']

ridgeModel = Ridge(alpha=0.2, normalize=False)

ridgeModel.fit(X,y)

ridgeModel.intercept_,ridgeModel.coef_


# Let's see how different this performs compared to the usual Linear Regression model. We would like to test if we can reduce the MSE on unseen test data if we use a non-zero shrinkage factor. This will demonstrate that the solution to the Ridge Regression line is less effected by a change in the data.
# 
# First Linear Regression

# In[78]:


# Run through each model in the correct order and run CV on it and save the best CV score
bestMeanCV = -1
bestMeanCVModel = []
oldArraySize = 0

X = housePrice.drop('SalePrice',axis=1)

columnsArray = X.columns.copy()

# y is the response variable
y = housePrice['SalePrice']

while oldArraySize != len(X):
    bestPredictor = ''
    oldArraySize = len(X.columns)
    for i in columnsArray:
        thisModel = bestMeanCVModel.copy()
        thisModel.append(i)
        # First set X to be the full set of remaining parameters
        x = X.loc[:,thisModel]

        if len(x.columns) == 1:
            linregCVScores = cross_val_score(Ridge(alpha=0.0001),x.values.reshape(-1,1),y,scoring='neg_mean_squared_error',cv=10)
        else:
            linregCVScores = cross_val_score(Ridge(alpha=0.0001),x,y,scoring='neg_mean_squared_error',cv=10)
            
        if bestMeanCV > -linregCVScores.mean():
            bestMeanCV = -linregCVScores.mean()
            bestPredictor = i
        elif bestMeanCV == -1:
            bestMeanCV = -linregCVScores.mean()
            bestPredictor = i
    
    if bestPredictor not in columnsArray:
        break
    
    columnsArray = columnsArray.drop(bestPredictor)
    bestMeanCVModel.append(bestPredictor)
    print('{} was added with test MSE {}'.format(bestMeanCVModel[-1],bestMeanCV))

        
print('The final best model is {} and its TEST MSE is {}'.format(bestMeanCVModel,bestMeanCV))


# Now with a larger $\lambda$

# In[79]:


# Run through each model in the correct order and run CV on it and save the best CV score
bestMeanCV = -1
bestMeanCVModel = []
oldArraySize = 0

X = housePrice.drop('SalePrice',axis=1)

columnsArray = X.columns.copy()

# y is the response variable
y = housePrice['SalePrice']

while oldArraySize != len(X):
    bestPredictor = ''
    oldArraySize = len(X.columns)
    for i in columnsArray:
        thisModel = bestMeanCVModel.copy()
        thisModel.append(i)
        # First set X to be the full set of remaining parameters
        x = X.loc[:,thisModel]

        if len(x.columns) == 1:
            linregCVScores = cross_val_score(Ridge(alpha=6),x.values.reshape(-1,1),y,scoring='neg_mean_squared_error',cv=10)
        else:
            linregCVScores = cross_val_score(Ridge(alpha=6),x,y,scoring='neg_mean_squared_error',cv=10)
            
        if bestMeanCV > -linregCVScores.mean():
            bestMeanCV = -linregCVScores.mean()
            bestPredictor = i
        elif bestMeanCV == -1:
            bestMeanCV = -linregCVScores.mean()
            bestPredictor = i
    
    if bestPredictor not in columnsArray:
        break
    
    columnsArray = columnsArray.drop(bestPredictor)
    bestMeanCVModel.append(bestPredictor)
    print('{} was added with test MSE {}'.format(bestMeanCVModel[-1],bestMeanCV))

        
print('The final best model is {} and its TEST MSE is {}'.format(bestMeanCVModel,bestMeanCV))


# A small improvement in the test MSE.

# In[80]:


X = housePrice.drop('SalePrice',axis=1)[bestMeanCVModel]

columnsArray = X.columns.copy()

# y is the response variable
y = housePrice['SalePrice']

rm = Ridge(alpha=6)
rm.fit(X,y)

s = 'SalePrice = {}'.format(round(rm.intercept_,2))

for i,j in zip(rm.coef_,bestMeanCVModel):
    s = s + ' + {}*{}'.format(round(i,2),j)
    
s


# ## Appendix

# ### A1 - $(2n) (2 \sum_{i=1}^n x_i^2) - (2 \sum_{i=1}^n x_i)^2 > 0$ for non-trivial X
# 
# Statement: $(2n) (2 \sum_{i=1}^n x_i^2) - (2 \sum_{i=1}^n x_i)^2 > 0 \; \forall \; n > 1$
# 
# Proof: We prove this by induction on $n$. If n = 1, we have $(2n) (2 \sum_{i=1}^n x_i^2) - (2 \sum_{i=1}^n x_i)^2 = 0$, but this is not what we want.
# 
# Let n = 2 > 1. Then 
# 
# $$(2n) (2 \sum_{i=1}^n x_i^2) - (2 \sum_{i=1}^n x_i)^2 = 2 x_1^2 + 2 x_2^2 - (x_1 + x_2)^2$$
# 
# $$= 2 x_1^2 + 2 x_2^2 - x_1^2 - x_2^2 - 2x_1 x_2 = x_1^2 + x_2^2 - 2x_1 x_2 = (x_1 - x_2)^2  > 0$$
# 
# So we have proved the assertion for n = 2.
# 
# Let us prove the statement for n+1 assuming it is true for n.
# 
# i.e. Assume $n \sum_{i=1}^n x_i^2 - (\sum_{i=1}^n x_i)^2 > 0$
# 
# Then 
# 
# $$(n+1) \sum_{i=1}^{n+1} x_i^2 - (\sum_{i=1}^{n+1} x_i)^2 = (n+1)[\sum_{i=1}^{n} x_i^2 + x_{n+1}^2] - (\sum_{i=1}^{n} x_i + x_{n+1})^2$$
# 
# $$= [n \sum_{i=1}^n x_i^2 + \sum_{i=1}^n x_i^2 + (n+1)x_{n+1}^2] - (\sum_{i=1}^n x_i)^2 - x_{n+1}^2 + 2x_{n+1} \sum_{i=1}^n x_i$$
# 
# $$= n \sum_{i=1}^n x_i^2 - (\sum_{i=1}^n x_i)^2 + \sum_{i=1}^n x_i^2 + (n+1)x_{n+1}^2 - x_{n+1}^2 + 2x_{n+1} \sum_{i=1}^n x_i$$
# 
# by the assumption for n we have
# 
# $$> \sum_{i=1}^n x_i^2 + (n+1)x_{n+1}^2 - x_{n+1}^2 + 2x_{n+1} \sum_{i=1}^n x_i$$
# 
# by the assumption for n that $\sum_{i=1}^n x_i^2 > \frac{1}{n}(\sum_{i=1}^n x_i)^2$ we have
# 
# $$> \frac{1}{n}(\sum_{i=1}^n x_i)^2 + (n+1)x_{n+1}^2 - x_{n+1}^2 + 2x_{n+1} \sum_{i=1}^n x_i =\frac{1}{n}(\sum_{i=1}^n x_i)^2 + nx_{n+1}^2 + 2x_{n+1} \sum_{i=1}^n x_i $$
# 
# $$= \frac{1}{n}[(\sum_{i=1}^n x_i)^2 + n^2 x_{n+1}^2 + 2n x_{n+1} \sum_{i=1}^n x_i]$$
# 
# $$=\frac{1}{n}\left( \sum_{i=1}^n x_i + n x_{n+1} \right)^2 > 0$$
# 
# This proves the statement. This assumes that at least one $X_i$ is non-zero.

# ### A2 - Maximum Likelihood Estimation (MLE)
# 
# Let's assume that there is a linear relationship between the response and predictor variables and that any discrepency is due to random noise, this is expressed as
# 
# $$Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$$
# 
# where the errors are normally distributed, $\epsilon \sim N(0,\sigma^2)$. Then, the response variable given the data are normally distributed
# 
# $$Y_i|X_i \sim N(\beta_0 + \beta_1 X_i,\sigma^2)$$
# 
# where the mean or expectation is
# 
# $$E \left[  Y_i|X_i  \right] = E[\beta_0 + \beta_1 X_i + \epsilon_i] = E[\beta_0] + E[\beta_1 X_i] + E[\epsilon_i] = \beta_0 + \beta_1 X_i$$
# 
# The probability density function for $Y_i$ is then
# 
# $$P(Y_i=y_i|X_i) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2\sigma^2} [y_i - (\beta_0 + \beta_1 x_i) ]^2 \right)$$
# 
# Then, if the $Y_i$ observations are independent of each other, we have that the likelihood of $\beta = (\beta_0,\beta_1)$ (the probability of observing this data given these parameters) is
# 
# $$L(\beta) = P(Y|\beta,X) = P(Y_1=y_1,Y_2=y_2,...,Y_n=y_n|\beta,X) = P(Y_1=y_1|\beta,X_1)P(Y_2=y_2|\beta,X_2)...,P(Y_n=y_n|\beta,X_n)$$
# 
# where the last equality is due to the independence of each observation and that $Y_i$ is only dependent on $\beta$ and $X_i$. Using the probability density function above, this becomes
# 
# $$L(\beta) = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2\sigma^2} [y_i - (\beta_0 + \beta_1 x_i) ]^2 \right) = \left( \frac{1}{\sqrt{2 \pi \sigma^2}} \right)^n \exp \left( -\frac{1}{2\sigma^2} \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2 \right)$$
# 
# Therefore, maximising this function with respect to $\beta$, corresponds to finding values for $\beta$ which maximises the probability of obtaining this response data given the predictor data. Instead of working with this equation as it stands, we note that the right hand side of the above equation is positive for all values of $\beta$ and $x_i$. This means that we can apply a handy trick in that since the $\log$ function is a monotonically increasing function, maximising $\log(L(\beta))$ is the same as maximising $L(\beta)$. Due to the existence of $\exp$ in $L(\beta)$, we may choose the natural logarithm so that the exponential disappears (we will still denote this as $\log$).
# 
# $$l(\beta) = \log(L(\beta)) = \log \left(\left( \frac{1}{\sqrt{2 \pi \sigma^2}} \right)^n \right) -\frac{1}{2\sigma^2} \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2$$
# 
# Since the first term on the right-hand side is indifferent to the choice of $\beta$, maximising $l(\beta)$ corresponds to maximising the last term on the right-hand side
# 
# $$\max_{\beta} l(\beta) = \max_{\beta} \left( - \frac{1}{2\sigma^2} \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2 \right)$$
# 
# which is equivalent to
# 
# $$\max_{\beta} l(\beta) = \min_{\beta} \left( \sum_{i=1}^n [y_i - (\beta_0 + \beta_1 x_i) ]^2 \right) = \min_{\beta} RSS$$
# 
# where $RSS = \sum_{i=1}^n \epsilon_i^2$. Note that for multiple predictors ($p$ predictors), the above becomes
# 
# $$\max_{\beta} l(\beta) = \min_{\beta} \left( \sum_{i=1}^n \left[y_i - \left(\beta_0 + \sum_{j=1}^p \beta_j x_{ij} \right) \right]^2 \right) = \min_{\beta} RSS$$
# 
# where $x_{ij}$ is the $j^{th}$ predictor for observation $i$.

# ### A3 - The mean point ($\bar{X}$,$\bar{Y}$) lies on the linear regression line
# 
# Let's assume that the random variable that represents the response be assumed to be linearly dependent on the predictors:
# 
# $$Y = \beta_0 + \beta_1 X + \epsilon$$
# 
# We approximate the coefficients using the data we have observed:
# 
# $$\hat{Y} = \hat{\beta_0} + \hat{\beta_1} X$$
# 
# Note that it is assumed that $\beta_i$ and $\hat{\beta}_i$ are constant and determined such that they satisfy the line of best fit. Taking the expectation of both sides of the above equations:
# 
# $$\mu_Y = E[Y] = E[\beta_0 + \beta_1 X + \epsilon] = E[\beta_0] + E[\beta_1 X] + E[\epsilon] = \beta_0 + \beta_1 E[X] + 0 = \beta_0 + \beta_1 \mu_X$$
# 
# $$\mu_{\hat{Y}} = E[\hat{Y}] = E[\hat{\beta_0} + \hat{\beta_1} X] = E[\hat{\beta_0}] + E[\hat{\beta_1} X] = \hat{\beta_0} + \hat{\beta_1} E[X] + 0 = \hat{\beta_0} + \hat{\beta_1} \mu_{\hat{X}}$$
# 
# The first equation above says that if we assume the linear model, then the population means $(\mu_X,\mu_Y)$ must be a solution to this model. The second equation says that the point $(\mu_{\hat{X}},\mu_{\hat{Y}})$ must lie on any linear model we fit to the data regardless of the coefficients we have chosen. Now the sample means are easily obtained and have the exact equality below:
# 
# $$\mu_{\hat{Y}} = \bar{Y}$$
# $$\mu_{\hat{X}} = \bar{X}$$
# 
# This result also holds when $\boldsymbol{X}$ is a vector of predictors.

# ### A4 -  For a single predictor, $R^2 = Cor(X,Y)^2$
# 
# We start with the definition of $R^2$:
# 
# $$R^2 = \frac{ TSS - RSS }{ RSS }$$
# 
# Using $TSS = \sum_{i=1}^n (y_i - \bar{y})^2$ and $RSS = \sum_{i=1}^n (y_i - \hat{y})^2$
# 
# $$R^2 = \frac{ \sum_{i=1}^n (y_i - \bar{y})^2 - \sum_{i=1}^n (y_i - \hat{y})^2 }{ \sum_{i=1}^n (y_i - \hat{y})^2 } = \frac{ \sum_{i=1}^n [ (y_i - \bar{y}) - (y_i - \hat{y}) ][ (y_i - \bar{y}) + (y_i - \hat{y}) ] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# $$= \frac{ \sum_{i=1}^n [ (y_i - \bar{y}) - (y_i - \hat{y}) ][ (y_i - \bar{y}) + (y_i - \hat{y}) ] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# Using $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$ and $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i = \bar{y} - \hat{\beta}_1 \bar{x} + \hat{\beta}_1 x_i = \bar{y} -\hat{\beta}_1(\bar{x} - x_i)$
# 
# $$R^2= \frac{ \sum_{i=1}^n [ (y_i - \bar{y}) - (y_i - \bar{y} -\hat{\beta}_1(\bar{x} - x_i)) ][ (y_i - \bar{y}) + (y_i - \bar{y} -\hat{\beta}_1(\bar{x} - x_i)) ] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# $$= \frac{ \sum_{i=1}^n [ \hat{\beta}_1(\bar{x} - x_i) ][ 2(y_i - \bar{y}) -\hat{\beta}_1(\bar{x} - x_i) ] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# $$= \frac{ \hat{\beta}_1 \left[ 2 \sum_{i=1}^n (\bar{x} - x_i)(y_i - \bar{y}) - \hat{\beta}_1\sum_{i=1}^n (\bar{x} - x_i)^2 \right] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# Using $\hat{\beta}_1 = \frac{ \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) }{ \sum_{i=1}^n (x_i - \bar{x})^2 }$
# 
# $$R^2 = \frac{ \hat{\beta}_1 \left[ 2 \sum_{i=1}^n (\bar{x} - x_i)(y_i - \bar{y}) - \sum_{i=1}^n (\bar{x} - x_i)(y_i - \bar{y}) \right] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# $$= \frac{ \hat{\beta}_1 \left[ \sum_{i=1}^n (\bar{x} - x_i)(y_i - \bar{y}) \right] }{ \sum_{i=1}^n (y_i - \bar{y})^2 }$$
# 
# $$= \frac{ \left[ \sum_{i=1}^n (\bar{x} - x_i)(y_i - \bar{y}) \right]^2 }{ \sum_{i=1}^n (y_i - \bar{y})^2 \sum_{i=1}^n (x_i - \bar{x})^2 }$$
# 
# $$= \frac{ \left[ \sum_{i=1}^n (\bar{x} - x_i)(y_i - \bar{y}) \right]^2 }{ \left[ \sqrt{ \sum_{i=1}^n (y_i - \bar{y})^2 \sum_{i=1}^n (x_i - \bar{x})^2} \right]^2}$$
# 
# $$= corr(X,Y)^2$$

# ### A5 - Variance of $\beta_{0}$ and $\beta_{1}$

# First, note that $y$ is a dependent variable on $x$. This means that any linear model and subsequently the calculations of $\beta_{0}$ and $\beta_{1}$ are susceptible to a variation of $y$ for a given $x$ value. Hence in the derivation of the variance of those parameters $x$ values are treated as a constant.
# 
# We start with the definition of $\beta_{1}$:
#     $$\beta_{1}=\frac{\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}}$$
#     
# The variance of $\beta_{1}$ is therefore given by:
# $$Var(\beta_{1})=Var\left[\frac{\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}}\right]\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}Var\bigg[\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{x})\bigg]\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}Var\bigg[\sum_{i=1}^{n}(x_{i}y_{i}-x_{i}\bar{y}-\bar{x}y_{i}+\bar{x}\bar{y})\bigg]\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}Var\bigg[\sum_{i=1}^{n} x_{i}y_{i}-\sum_{i=1}^{n} x_{i}\bar{y}-\sum_{i=1}^{n} \bar{x}y_{i}+\sum_{i=1}^{n} \bar{x}\bar{y}\bigg]\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}Var\bigg[\sum_{i=1}^{n} x_{i}y_{i}- n \bar{x}\bar{y}-\sum_{i=1}^{n} \bar{x}y_{i}+ n \bar{x}\bar{y}\bigg]\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}Var\bigg[\sum_{i=1}^{n}(x_{i}y_{i}-\bar{x}y_{i})\bigg]$$
# 
# As each observation is indepedent from another ($y_{i}$ are independent of each other) we have:
# 
# $$Var(\beta_{1})=\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}\sum_{i=1}^{n}Var(x_{i}y_{i}-\bar{x}y_{i})\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}Var(y_{i})\\
# =\frac{1}{\left(\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right)^{2}}\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\sigma_{y}^{2}\\
# =\frac{\sigma_{y}^{2}}{\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}}$$
# 
# However since $y_{i}=\beta_{0}+\beta_{1}x_{i}+\epsilon_{i}$ and $\epsilon_{i}$ is the only random variable on the right hand side, we have:
# 
# $$Var(y_{i})=Var(\epsilon_{i})=\sigma^{2}$$. 
# 
# Then our expression above becomes:
# 
# $$Var(\beta_{1})= \frac{\sigma^{2}}{\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}}  $$
# 
# Since $\beta_{0}=\bar{y}-\beta_{1}\bar{x}$ we have:
# 
# \begin{equation}
# \begin{split}
# E(\beta_0) &= E[\bar{y}] + \bar{x}E[\beta_1]\\
# &= E[\frac{1}{n} \sum_{i=1}^n y_i] + \bar{x}\frac{1}{\sum_{k=1}^n (x_i - \bar{x})^2}E[\sum_{i=1}^n x_i y_i - n\bar{x}\bar{y}]\\
# &= \mu_Y + \bar{x}\frac{1}{\sum_{k=1}^n (x_i - \bar{x})^2}[\sum_{i=1}^n x_i E[y_i] - n\bar{x}E[\bar{y}]]\\
# &= \mu_Y + \bar{x}\frac{1}{\sum_{k=1}^n (x_i - \bar{x})^2}[\mu_Y \sum_{i=1}^n x_i - n\bar{x}\mu_Y]\\
# &= \mu_Y
# \end{split}
# \end{equation}
# 
# and
# 
# \begin{equation}
# \begin{split}
# E(\beta_0^2) &= E[\bar{y}^2 + 2 \beta_1 \bar{x} \bar{y} + \beta_1^2 \bar{x}^2]\\
# &= E[\bar{y}^2] + 2 \bar{x} E[\beta_1 \bar{y}] + \bar{x}^2 E[\beta_1^2]\\
# &= Var(\bar{y}) + E[\bar{y}]^2 + \bar{x}^2 \left[ Var(\beta_1) + E[\beta_1]^2  \right]\\
# &= Var(\bar{y}) + \mu_Y^2 + \bar{x}^2 \left[ Var(\beta_1) \right]\\
# &= \frac{\sigma^2}{n} + \mu_Y^2 + \frac{\bar{x}^2 \sigma^2}{\sum_{k=1}^n (x_k - \bar{x})}
# \end{split}
# \end{equation}
# 
# finally
# 
# \begin{equation}
# \begin{split}
# Var(\beta_0) &= E[\beta_0^2] - E[\beta_0]^2\\
# &= \sigma^{2}\left[ \frac{1}{n} + \frac{\bar{x}^{2}}{\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}}  \right]
# \end{split}
# \end{equation}
# 
# Note that, with a bit of algebraic manipulation $\big($hint: $\sum(x_{i}-\bar{x})^{2}=\sum x_{i}^{2}-n\bar{x}^{2})$$\big)$, this is also equal to:
# 
# $$Var(\beta_{0}) = \frac{\sigma^{2}\sum_{i=1}^{n}x_{i}^{2}}{n\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2} } $$
