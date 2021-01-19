# Clemence Le Cornec
# 07.01.2021

import pandas as pd
import numpy as np
import os
import random
from scipy.optimize import minimize
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

class REVS(object):
	"""REVS model"""
	
	def __init__(self, pathData, vehicleTechnologyCategory, nbJacknife, minSpeed, maxSpeed, speedBin, VSPBin):
		"""Initialisation of the class. nbJacknife corresponds to the number of iterations performed to train the model for one vehicle techonology category"""
		"""Data for each individual vehicle technology category is stored in the same folder"""
		self.pathData = pathData
		self.vehicleTechnologyCategory = vehicleTechnologyCategory
		self.nbJacknife = nbJacknife
		self.minSpeed = minSpeed
		self.maxSpeed = maxSpeed
		self.speedBin = speedBin
		self.VSPBin = VSPBin
		
	def loadData(self, pathData):
		"""Function to load all the data for one defined vehicle technology category"""
		"""IN addition of the average speed, average VSP and NOx EF, a column records the ID of each individual vehicle"""
		finalData = []
		filenames =  [f for f in os.listdir(self.pathData)]
		
		for filename in filenames:
			if len(finalData) == 0:
				finalData = np.load(self.pathData +"/"+ filename)
				finalData = np.hstack((finalData, int(filename.split("_")[0]) * np.ones((len(finalData),1)))) # Use the vehicle ID to identify the vehicle
			else:
				data = np.load(self.pathData +"/"+ filename)
				data = np.hstack((data, int(filename.split("_")[0]) * np.ones((len(data),1))))
				finalData = np.vstack((finalData, data))
		self.finalData = finalData
	
	def evaluateREVS(self, coefficients, x):
		"""Evaluate the emissions based on the REVS model"""
		alpha, beta = coefficients
		return np.exp(alpha * x + beta)
	
	def logLikelihoodREVS(self, coefficients, x, y):
		"""Log likelihood of the REVS model"""
		REVS = self.evaluateREVS(coefficients, x)
		return -np.sqrt(np.sum((y - REVS) ** 2))
		
	def fitREVS(self, minMaxVSP, statistics, maxNOx):
		"""By default, the estimation of the REVS coefficients is performed in a non biased way (each vehicle is given a relative weight of 1)"""
		"""minMaxVSP corresponds to the VSP limits by range of speed (depends on the type of data and driving style considered)"""
		"""make sure there are enough datapoints per category to allow training to take place properly"""
		
		REVSCoefficients = np.zeros((len(np.arange(self.minSpeed, self.maxSpeed, self.speedBin)), 4))
		
		alpha0 = 0.1
		beta0 = 0.1
		
		for j in range(self.minSpeed, self.maxSpeed, self.speedBin):
			coefficients = []
			for k in range(0, self.nbJacknife):
				meanSpeed = []
				meanVSP = []
				meanNOxEF = []
				stdNOxEF = []
				if (j>= 10) & (j < 50):
					minVSP = minMaxVSP[0][0]
					maxVSP = minMaxVSP[0][1]
				elif (j>= 50) & (j < 90):
					minVSP = minMaxVSP[1][0]
					maxVSP = minMaxVSP[1][1]
				elif j >= 90:
					minVSP = minMaxVSP[2][0]
					maxVSP = minMaxVSP[2][1]
				for l in np.arange(minVSP, maxVSP, self.VSPBin):
					indices = np.where((self.finalData[:, 0] >= j) & (self.finalData[:,0] < (j + self.speedBin)) & (self.finalData[:,1] >= l) &  (self.finalData[:,1] != 0) & (self.finalData[:,1] < (l + self.VSPBin)) &  (self.finalData[:,2] > 0) & (self.finalData[:,2] < maxNOx) )[0] 
					if len(indices) > 9:
						# Select randomly 80% of the data
						indicesFinal = random.sample(list(indices),len(indices))[0:int(0.80*len(indices))]
						# Compute the relative weighing of each vehicle
						dataframe = pd.DataFrame(self.finalData[indicesFinal,:], columns = ["speed", "VSP", "NOx", "ID"])
						weights = 1 / dataframe.groupby(["ID"])["ID"].transform("count").values
						# Compute the non biased mean and standard deviation
						meanSpeed.append(np.average(self.finalData[indicesFinal,0], weights = weights))
						meanVSP.append(np.average(self.finalData[indicesFinal,1], weights = weights))
						meanNOxEF.append(np.average(self.finalData[indicesFinal,2], weights = weights))
						stdNOxEF.append((np.sqrt(np.average((self.finalData[indicesFinal,2]-np.average(self.finalData[indicesFinal,2], weights = weights))**2, weights=weights))))
				
				# average NOx emission factor
				nll = lambda *args: -self.logLikelihoodREVS(*args)
				initial = np.array([alpha0, beta0]) + 0.015 * np.random.randn(2) 
				soln = minimize(nll, initial, args=(np.asarray(meanVSP), np.asarray(meanNOxEF)))
				alpha,beta = soln.x
				
				# standard deviation of the average NOx emission factor
				nll = lambda *args: -self.logLikelihoodREVS(*args)
				initial = np.array([alpha0, beta0]) + 0.015 * np.random.randn(2)
				soln = minimize(nll, initial, args=(np.asarray(meanVSP), np.asarray(stdNOxEF))) 
				gamma, delta = soln.x
				
				coefficients.append([alpha,beta,gamma,delta])
			
			# Compute the maximum likelihood of the coefficients using chain consumer
			cc = ChainConsumer()
			cc.add_chain(np.asarray(coefficients)[:,0:2], parameters=["alpha", "beta"]).configure(statistics=statistics, summary_area=0.95)
			dataFinal = cc.analysis.get_summary()
			alpha = dataFinal["alpha"][1]
			beta = dataFinal["beta"][1]
			
			cc = ChainConsumer()
			cc.add_chain(np.asarray(coefficients)[:,2:4], parameters=["gamma", "delta"]).configure(statistics=statistics, summary_area=0.95)
			dataFinal = cc.analysis.get_summary()
			gamma = dataFinal["gamma"][1]
			delta = dataFinal["delta"][1]
			
			# Save the final coefficients
			REVSCoefficients[int((j - self.minSpeed)/self.speedBin), :] = [alpha, beta, gamma, delta]
	
		self.REVSCoefficients = REVSCoefficients
	
	def printREVSCoefficients(self):
		"""Print REVS coefficients on the command line"""
		print(self.REVSCoefficients)
	
	def saveREVSCoefficients(self, pathSaveData, nameFile):
		"""Save REVS into a numpy array"""
		np.save(pathSaveData + "%s.npy"%(nameFile), self.REVSCoefficients)
	
	def loadREVSCoefficients(self, pathSaveData, nameFile):
		""" Load REVSCoefficients from numpy array"""
		self.REVSCoefficients = np.load(pathSaveData + "%s.npy"%(nameFile))
		
	def predictEmissions(self, averageVSP, averageSpeed):
		""" Predict the emissions based on the REVS model """
		""" The coefficients of the REVS model should be loaded beforehand"""
		""" averageSpeed is given in km/h and averageVSP in kW/t. These are vectors"""
		coefficientsIndices = int(np.floor((averageSpeed - self.minSpeed)/self.speedBin))
		averageEF = self.evaluateREVS(self.REVSCoefficients[coefficientsIndices,0:2], averageVSP)
		stdEF = self.evaluateREVS(self.REVSCoefficients[coefficientsIndices,2:4], averageVSP)
		return averageEF, stdEF
		
	def plotREVSPerformance(self, minMaxVSP, maxY):
		"""Plot the performances of REVS against the data"""
		
		count = 1
		xTicksFinal = []
		xTickLabelsFinal = []
		fig = plt.figure(figsize = (14,14))
		for j in range(self.minSpeed, self.maxSpeed, self.speedBin):
			averageEF = []
			stdEF = []
			xTicks = []
			if (j>= 10) & (j < 50):
				minVSP = minMaxVSP[0][0]
				maxVSP = minMaxVSP[0][1]
			elif (j>= 50) & (j < 90):
				minVSP = minMaxVSP[1][0]
				maxVSP = minMaxVSP[1][1]
			elif j >= 90:
				minVSP = minMaxVSP[2][0]
				maxVSP = minMaxVSP[2][1]
			for l in np.arange(minVSP, maxVSP, self.VSPBin):
				indices = np.where((self.finalData[:, 0] >= j) & (self.finalData[:,0] < (j + self.speedBin)) & (self.finalData[:,1] >= l) & (self.finalData[:,1] < (l + self.VSPBin)) & (self.finalData[:,2] > 0) )[0] 
				print(j, l, len(indices))
				if len(indices) > 9:
					averageSpeed = np.nanmean(self.finalData[indices, 0])
					averageVSP = np.nanmean(self.finalData[indices, 1])
					coefficientsIndices = int(np.floor((averageSpeed - self.minSpeed)/self.speedBin))
					averageEF.append(self.evaluateREVS(self.REVSCoefficients[coefficientsIndices,0:2], averageVSP))
					stdEF.append(self.evaluateREVS(self.REVSCoefficients[coefficientsIndices,2:4], averageVSP))
					box = plt.boxplot(self.finalData[indices, 2], positions = [count], widths = 0.6, showmeans = True,  meanprops={"markerfacecolor":"k", "markeredgecolor":"k"}, flierprops = dict(marker='o', markerfacecolor="k", markeredgecolor="k"))
					for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
						plt.setp(box[element], color="k")
					xTicks.append(count)
					xTicksFinal.append(count)
					xTickLabelsFinal.append("%s "%(str(l)))
					count += 1
			count += 2
			plt.plot(xTicks, averageEF, linestyle = "--", color = "blue", alpha = 0.8)
			plt.fill_between(xTicks, np.asarray(averageEF)-np.asarray(stdEF), np.asarray(averageEF)+np.asarray(stdEF), color = "blue", alpha = 0.4)
		plt.xticks(xTicksFinal, labels = xTickLabelsFinal)
		plt.xlabel("VSP [kW/t]")
		plt.ylabel("NO$_\mathrm{x}$ EF [g/km]")
		plt.ylim([0, maxY])
		
		plt.show()
		
		
