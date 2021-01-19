# Clemence Le Cornec
# 07.01.2021

from Vehicles import *
import numpy as np
import os

pathSaveData = os.getcwd() + "/DataFinal/ExtractedData/"
pathData = os.getcwd() + "/DataFinal/RawData/"
type = "urban" # Can be urban, rural or motorway
distanceExtracted = 0.5

folders = [f for f in os.listdir(pathData)]

for folder in folders:
	
	pathFinal = pathData + folder + "/"
	
	foldersData = [f for f in os.listdir(pathFinal)]
	
	for vehicleData in foldersData:

		# Read the text file containing the data
		specification = open(pathFinal + vehicleData + "/specification.txt", "r")
		lines = specification.readlines()
		specData = []
		for line in lines:
			specData.append(line.split(":")[1].split("\n")[0].replace(" ", ""))
		specification.close()
		
		# Create the object and extract the data needed
		veh = Vehicle(ID = specData[0], fuel = specData[3], manufacturer = specData[1], model = specData[2], euroStandard = specData[4], height = float(specData[5]),\
									width = float(specData[6]), weight = float(specData[7]), pathData = pathFinal + vehicleData + "/")
		# Load the data and save the extracted data
		veh.loadTestData(nameColumns = ["altitude", "speed", "Nox"], type = type, distanceSegment = distanceExtracted, pathSaveSegments = pathSaveData + folder + "/")