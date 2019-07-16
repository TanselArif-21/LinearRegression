# Import modules
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from scipy import stats





class RidgeRegression(Ridge):
	
	def summary(self,X,y):
		# Get the coefficient solutions from the model
		coefs = []
		
		# If the model was fit with an intercept
		if 'intercept_' in dir(self):
			coefs = np.append(self.intercept_,self.coef_)
		else:
			coefs = self.coef_

		# Get the predictions (X_new includes the constant term)
		predictions = self.predict(X)

		# If a constant column needs to be added
		if len(X.columns) < len(coefs):
			X = X.copy()
			X.insert(0,'Const',1)
		
		# Calculate the MSE
		MSE = (sum((y-predictions)**2))/(len(X)-len(X.columns))

		# Calculate the variance
		var = MSE*(np.linalg.inv(np.dot(X.T,X)).diagonal())

		# Calculate the standard deviation
		sd = np.sqrt(var)

		# Calculate the t-statistics
		t = coefs/ sd

		# Calculate the p-values using the t-statistics and the t-distribution (2 is two-sided)
		p_values =[2*(1-stats.t.cdf(np.abs(i),(len(X)-1))) for i in t]

		# 3 decimal places to match statsmodels output
		var = np.round(var,3)
		t = np.round(t,3)
		p_values = np.round(p_values,3)

		# 4 decimal places to match statsmodels
		coefs = np.round(coefs,4)
		
		# Summary dataframe
		summary_df = pd.DataFrame()
		summary_df["Features"],summary_df["coef"],summary_df["std err"],summary_df["t"],summary_df["P > |t|"] = [X.columns,
																									coefs,sd,t,p_values]
		print(summary_df)  

	def featureSelection(self,X,y):
		# Run through each model in the correct order and run CV on it and save the best CV score
		bestMeanCV = -1
		bestMeanCVModel = []
		oldArraySize = 0

		columnsArray = X.columns.copy()

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

		
		self.bestMeanCVModel = bestMeanCVModel
		self.bestMeanCV = bestMeanCV
		print('The final best model is {} and its TEST MSE is {}'.format(bestMeanCVModel,bestMeanCV))