# Clemence Le Cornec
# 07.01.2021

import numpy as np
import matplotlib.pyplot as plt
import random

def computeCoefficientsVSP(width, height, weight, fuel):
	"""Compute the coefficients used to compute VSP. It follows the methodology developped in Davison et al., 2020, Distance-based emission factors from vehicle emissions
		remote sensing measurements"""
	
	area = width * height # compute the frontal area
	weight = weight / 1000 # in tons
	
	if area < 3:
	
		if weight  < 1.3:
			
			if fuel == "Diesel":
				R0 = 120
				R1 = 0.77
				CdA = 0.537
			else:
				R0 = 106
				R1 = 0.67
				CdA = 0.538
		
		else:
		
			if area < 2.7:
			
				if fuel == "Diesel":
					R0 = 151
					R1 = 0.93
					CdA = 0.617
				else:
					R0 = 139
					R1 = 0.85
					CdA = 0.618
			
			else:
				
				if fuel == "Diesel":
					R0 = 166
					R1 = 1.02
					CdA = 0.665
				else:
					R0 = 154
					R1 = 0.94
					CdA = 0.689
	
	else:
		
		if fuel == "Diesel":
			R0 = 204
			R1 = 1.18
			CdA = 0.915
		else:
			R0 = 175
			R1 = 1.01
			CdA = 0.810
	
	return R0, R1, CdA

def computeAcceleration(speed):
	"""Compute acceleration directly from speed"""
	"""Speed must be given in km/h"""
	acceleration = (speed[2:len(speed)]-speed[0:len(speed)-2])*(1000/3600)
	acceleration = np.hstack((0,acceleration,0))
	acceleration = acceleration / 2
	
	return acceleration
	
def computeSlope(altitude, speed):
	""""Compute slope from altitude"""
	distance = 1000 * np.cumsum(speed)/3600
	slope = (altitude[1:len(altitude)] - altitude[0:len(altitude)-1])/(distance[1:len(distance)] - distance[0:len(distance)-1])
	slope = np.hstack((0,slope))
	return slope
	
def computeVSPFinal(speed, acceleration, slope, weightVeh, R0, R1, CdA):
	"""Compute the instantaneous VSP from the speed, acceleration and slope and the road load coefficients"""
	"""Speed must be in km/h, acceleration in m/s2 and the slope in m/m and weightVeh in kg"""
	"""See Davison et al, 2020 for more information"""
	speed = speed / (3.6) # change to m/s
	VSP = ( (2500 + (R0 * speed + R1 * speed**2 + CdA * 0.5 * 1.2 * speed** 3) * 1.08) / weightVeh) + speed * 1.08 * (1.04 * acceleration + 9.81 * slope)
	
	return VSP