# Clemence Le Cornec
# 08.01.2021

from REVS import *
import numpy as np
import os
import matplotlib.pyplot as plt

# Parameters
vehicleTechnologyCategory = "DE5"
pathData = os.getcwd() + "/DataFinal/ExtractedData/" + vehicleTechnologyCategory
pathSaveData = os.getcwd() + "/CoefficientsFinal/"
pathPreTrainedData = os.getcwd() + "/CoefficientsTrained/"
statistics = "mean" # Can be ["max_shortest", "max_symmetric", "max", "cumulative"], please check the chainconsumer documentation
nbJacknife = 1000
minSpeed = 10
maxSpeed = 130
speedBin = 10 # Width of the bins of speed
VSPBin = 2 # Width of the bins of VSP
minMaxVSP = [[-2,6],[2,12],[6,18]] # VSP limits for urban, rural and motorway trips respectively
maxNOx = 10 # if it is needed to eliminate some outliers


# Create a modelpython3 
model = REVS(pathData, vehicleTechnologyCategory, nbJacknife, minSpeed, maxSpeed, speedBin, VSPBin)

# Load the data
model.loadData(pathData)

# Train the model
model.fitREVS(minMaxVSP, statistics, maxNOx)

# Save the coefficients
model.saveREVSCoefficients(pathSaveData, vehicleTechnologyCategory)

# Print coefficients
model.printREVSCoefficients()

# Load the coefficients
model. loadREVSCoefficients(pathPreTrainedData, vehicleTechnologyCategory)

# Plot results
model.plotREVSPerformance(minMaxVSP, 4)