# Clemence Le Cornec
# 07.01.2021

import numpy as np
import pandas as pd
import os
from utils import *
from operator import itemgetter
from itertools import groupby
import matplotlib.pyplot as plt

class Vehicle(object):
	"""Represent a vehicle object"""
	
	def __init__(self, ID, fuel, manufacturer, model, euroStandard, height, width, weight, pathData):
		"""Class initialisation"""
		""" It requests the tests from each individual vehicle to be store in separated folders"""
		""" Please note that the resolution is considered to be 1 second"""
		""" All the processing of the raw PEMS data must be conducted before"""
		self.ID = ID
		self.fuel = fuel
		self.manufacturer = manufacturer
		self.model = model
		self.EuroStandard = euroStandard
		self.dimension = [width, height, weight]
		self.pathData = pathData
		self.extractedData = []
	
	def displayData(self):
		"""Display the data available"""
		print(self.__dict__)
		
	def loadTestData(self, nameColumns, type, distanceSegment, pathSaveSegments):
		"""Load the PEMS data collected for each individual vehicle"""
		filenames =  [f for f in os.listdir(self.pathData) if not f.endswith(".txt")]
		
		# Read the test data
		list_dataframe = [pd.read_csv(self.pathData + filename, usecols = nameColumns).fillna(0) if filename.split(".")[-1] == "csv" else pd.read_excel(self.pathData + filename, usecols = nameColumns).fillna(0) for filename in filenames]
		
		# Compute the road load coefficients
		R0, R1, CdA = computeCoefficientsVSP(self.dimension[0], self.dimension[1], self.dimension[2], self.fuel)
		
		# Add the filename to each dataframe
		for dataframe, filename in zip(list_dataframe, filenames):
			dataframe["filename"] = filename
			# Compute the acceleration from the speed and add to the dataframe
			dataframe["acceleration"] = computeAcceleration(dataframe["speed"].values)
			# Compute the slope from the altitude
			dataframe["slope"] = computeSlope(dataframe["altitude"].values, dataframe["speed"].values)
			dataframe["slope"] = dataframe["slope"].fillna(0).replace(-np.inf,np.nan).replace(np.inf,np.nan).dropna().reset_index(drop = True) 
			# Compute VSP from speed, acceleration and slope and add to the dataframe
			dataframe["VSP"] = computeVSPFinal(dataframe["speed"].values, dataframe["acceleration"].values, dataframe["slope"].values, self.dimension[2], R0, R1, CdA)
			# indices = dataframe[(dataframe["VSP"] < -100) | (dataframe["VSP"] > 50)].index # Remove unrealistic values of VSP [DEPENDING ON THE PRECISION OF THE GPS]
			# dataframe = dataframe.drop(indices).reset_index(drop = True) 
			# Drop the rows with inf or nan
			dataframe = dataframe.dropna()
			dataframe = dataframe.reset_index(drop = True)
			# Extract and concatenate segments
			segments = self.extractSegments(dataframe, type, distanceSegment)
			self.concatenateSegments(dataframe, segments)
		
		# Save the extracted segment data
		self.saveSegments(type, distanceSegment, pathSaveSegments)
		
		# Concatenate the dataframes for more ease of use
		loadedData = pd.concat(list_dataframe, ignore_index = False) # ignore index is false because we want to be able to distinguish easily between the different tests for segment extraction
		
		# Store data
		self.data = loadedData
		
	def extractSegments(self, dataframe, type, distanceSegment):
		"""Extract the segments from the raw data. Speed should be given in km/h"""
		
		if type == "urban":
			minSpeed = 0
			maxSpeed = 50
		elif type == "rural":
			minSpeed = 50
			maxSpeed = 90
		elif type == "motorway":
			minSpeed = 90
			maxSpeed = 150
		elif type == "all":
			minSpeed = 0
			maxSpeed = 150
			
		segments = []
		
		# Find the indices that corresponds to urban driving
		indices = np.where((dataframe["speed"].astype(np.float64).values >= minSpeed) & (dataframe["speed"].astype(np.float64).values <= maxSpeed))[0]
		# To be sure to be above the distance required
		distanceSegment += 0.015
		
		# Find the elements that follow each others
		for k, g in groupby(enumerate(indices), lambda ix : ix[0] - ix[1]):
			templist = list(map(itemgetter(1), g))
			distance = np.cumsum(dataframe.loc[templist, "speed"].astype(np.float64).values)/3600
			# If the vehicle was driven on a reasonable distance, select the subsegments
			if distance[-1] >= distanceSegment:
				# Break into segments of the right length
				numberOfSegments = int(np.floor(distance[-1] / distanceSegment))
				for m in range(0,numberOfSegments):
					beginIndex = templist[0] + np.abs(distance - m*distanceSegment).argmin()
					endIndex = templist[0] + np.abs(distance - (m+1)*distanceSegment).argmin()
					segments.append([range(beginIndex, endIndex)]) #templist
			del(templist)
			
		return segments
		
	def concatenateSegments(self, dataframe, segments):
		"""Concatenate the segments extracted"""
		
		finalData = np.zeros((len(segments),3))
		
		for k in range(0,len(segments)):
			averageSpeed = np.nanmean(dataframe.loc[segments[k][0],"speed"].values)
			averageVSP = np.nanmean(dataframe.loc[segments[k][0],"VSP"].values)
			NOxEF = np.sum(dataframe.loc[segments[k][0], 'Nox'].astype(np.float64).values) / (np.sum(dataframe.loc[segments[k][0], 'speed'].astype(np.float64).values)/3600)
			finalData[k,0] = averageSpeed
			finalData[k,1] = averageVSP
			finalData[k,2] = NOxEF
		
		if len(self.extractedData) == 0:
			self.extractedData = finalData
		else:
			self.extractedData = np.vstack((self.extractedData, finalData))
		
	def saveSegments(self, type, distanceSegment, pathSaveSegments):
		"""Save the extracted segments as numpy array"""
		np.save(pathSaveSegments + "%s_%s_%s_%s_%s_%s_%s.npy"%(self.ID, self.manufacturer, self.model, self.fuel, self.EuroStandard,distanceSegment, type), self.extractedData)