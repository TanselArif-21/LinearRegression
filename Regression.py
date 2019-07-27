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
	'''
	This class inherits from the Ridge class in the sklearn package. It extends that class by
	adding the capability to produce p-values, run feature selection and find the best alpha
	which minimises the MSE
	'''
	
	def summary(self,X,y):
		'''
		This method produces a summary similar to the one produced by statsmodels.api.
		It includes the coefficients and their p-values in a summary table
		:param X: features array
		:param y: response array
		'''
		
		# This will store the coefficients of the model that has already been run
		coefs = []
		
		# If the model was fit with an intercept
		if 'intercept_' in dir(self):
			coefs = np.append(self.intercept_,self.coef_)
		else:
			coefs = self.coef_

		# Get the predictions
		predictions = self.predict(X)

		# If a constant column needs to be added (determine this dynamically)
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
		
	def findBestAlpha(self,X,y,cv=10,scoring='neg_mean_squared_error',silent=True):
		'''
		This method keeps changing alpha until the MSE is reduced as much as it can be reduced. This
		alpha selection depends on input datasets
		:param X: features array
		:param y: response array
		:param silent: if True, then progress is omitted
		'''

		silent = True
		alpha = 1
		prevAlpha = None
		bestMSE = None
		tol = 0.000001
		doublingMode = True
		
		# Here, we start by continuously doubling alpha until we get to a point where the MSE doesn't increase
		# Then, at this point, we switch to incrementing (or decrementing) by smaller amounts until the tolerance is reached
		while True:
			# Calculate the MSE using this alpha
			thisMSE = np.mean(cross_val_score(RidgeRegression(alpha=alpha),X,y,scoring=scoring,cv=cv))

			if not silent:
				print('alpha = {}\nbestScore = {}\nthisScore = {}\n#############'.format(alpha,bestMSE,thisMSE))

			# if doubling mode
			if doublingMode:
				# update bestMSE
				if (not bestMSE) or (bestMSE < thisMSE):
					bestMSE = thisMSE
				else:
					if not silent:
						print('Doubling Finished!!!!')
						
					# switch the mode and roll back alpha to the previous one
					doublingMode = False
					tempAlpha = prevAlpha
					prevAlpha = alpha
					alpha = tempAlpha
					continue

				# update alpha
				prevAlpha = alpha
				alpha = (alpha + 0.001)*2
			else:        
				# update alpha to |alpha-prevAlpha|/2 away from where it currently is (in either direction)
				ghostPoint = alpha + (alpha - prevAlpha)
				nextAlpha1 = (prevAlpha + alpha)/2
				nextAlpha2 = (alpha + ghostPoint)/2
				
				# The Ridge class has numerical issues when alpha is close to zero
				if(nextAlpha1 < 0.0001):
					nextAlpha1 = 0.0001
				if(nextAlpha2 < 0.0001):
					nextAlpha2 = 0.0001
					
				# Calculate the MSE on either side of alpha
				MSE1 = np.mean(cross_val_score(RidgeRegression(alpha=nextAlpha1),X,y,scoring=scoring,cv=cv))
				MSE2 = np.mean(cross_val_score(RidgeRegression(alpha=nextAlpha2),X,y,scoring=scoring,cv=cv))

				# Choose to MSE and the corresponding alpha of the one that is better
				if (MSE1 > MSE2) and (MSE1 > bestMSE) and (np.abs(prevAlpha - alpha) > tol):
					prevAlpha = alpha
					alpha = nextAlpha1
					bestMSE = MSE1
				elif (MSE2 > MSE1) and (MSE2 > bestMSE) and (np.abs(prevAlpha - alpha) > tol):
					prevAlpha = alpha
					alpha = nextAlpha2
					bestMSE = MSE2
				else:
					if (np.abs(prevAlpha - alpha) > tol):
						# pull prevAlpha closer to alpha
						prevAlpha = (prevAlpha + alpha)/2
					else:
						alpha = prevAlpha
						break

		self.alpha = alpha
		print('Ridge Regression {} = {}, best alpha = {}'.format(scoring,bestMSE,alpha))

	def featureSelection(self,X,y,scoring='neg_mean_squared_error',cv=10):
		'''
		This method iterates and adds a new feature to the features list in the
		order of best improvement of MSE
		:param X: features array
		:param y: response array
		'''
		
		# Run through each model in the correct order and run CV on it and save the best CV score
		bestMeanCV = None
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
					linregCVScores = cross_val_score(Ridge(alpha=self.alpha),x.values.reshape(-1,1),y,scoring=scoring,cv=cv)
				else:
					linregCVScores = cross_val_score(Ridge(alpha=self.alpha),x,y,scoring=scoring,cv=cv)

				if not bestMeanCV:
					bestMeanCV = linregCVScores.mean()
					bestPredictor = i
				elif bestMeanCV < linregCVScores.mean():
					bestMeanCV = linregCVScores.mean()
					bestPredictor = i

			if bestPredictor not in columnsArray:
				break

			columnsArray = columnsArray.drop(bestPredictor)
			bestMeanCVModel.append(bestPredictor)
			print('{} was added with test {} {}'.format(bestMeanCVModel[-1],scoring,bestMeanCV))


		self.bestMeanCVModel = bestMeanCVModel
		self.bestMeanCV = bestMeanCV
		print('The final best model is {} and its TEST {} is {}'.format(bestMeanCVModel,scoring,bestMeanCV))
