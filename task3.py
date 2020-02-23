#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:44:34 2018

@author: karan
"""
import numpy as np
import math
import cv2

def polarCoordinates(x, y, dL, thetas, mat):
    for theta in range(thetas.shape[0]):
        rho = int(x*math.cos(math.radians(thetas[theta])) + y*math.sin(math.radians(thetas[theta])))
        mat[rho+dL][theta] += 1

def hough_peaks(mat, nPeaks, nSize=11):
    peakIndex = []
    mat1 = np.copy(mat)
    for i in range(nPeaks):
        iX = np.argmax(mat1)
        idx_y, idx_x = np.unravel_index(iX, mat1.shape)         
        
        peakIndex.append([idx_y, idx_x])
        
        minX = max(0, idx_x - (nSize//2))
        maxX = min(idx_x + (nSize//2) + 1, mat.shape[1])
        minY = max(0, idx_y - (nSize//2))
        maxY = min(idx_y + (nSize//2) + 1, mat.shape[0])

        for x in range(minX, maxX):
            for y in range(minY, maxY):
                mat1[y][x] = 0
    return np.array(peakIndex), mat


def drawLines(img, pI, dL, thetas):
    for i in range(pI.shape[0]):
        theta = thetas[pI[i][1]]
        rho = pI[i][0]-dL
        
        a = math.cos(math.radians(theta))
        b = math.sin(math.radians(theta))
        
        x = a*rho
        y = b*rho
        
        x1 = int(x - 1000*(b))
        y1 = int(y + 1000*(a))
        x2 = int(x + 1000*(b))
        y2 = int(y - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

def Sobel(I):
    sobelMatX = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobelMatY = [[-1,-2,-1],[0,0,0],[1,2,1]]
    m, n = I.shape
    image = np.pad(I, (1,1), 'constant')
    sobelMat = np.zeros(I.shape)
    for i in range(1, m):
        for j in range(1, n):        
            gx = (sobelMatX[0][0] * image[i-1][j-1]) + (sobelMatX[0][1] * image[i-1][j]) + \
                 (sobelMatX[0][2] * image[i-1][j+1]) + (sobelMatX[1][0] * image[i][j-1]) + \
                 (sobelMatX[1][1] * image[i][j]) + (sobelMatX[1][2] * image[i][j+1]) + \
                 (sobelMatX[2][0] * image[i+1][j-1]) + (sobelMatX[2][1] * image[i+1][j]) + \
                 (sobelMatX[2][2] * image[i+1][j+1])
    
            gy = (sobelMatY[0][0] * image[i-1][j-1]) + (sobelMatY[0][1] * image[i-1][j]) + \
                 (sobelMatY[0][2] * image[i-1][j+1]) + (sobelMatY[1][0] * image[i][j-1]) + \
                 (sobelMatY[1][1] * image[i][j]) + (sobelMatY[1][2] * image[i][j+1]) + \
                 (sobelMatY[2][0] * image[i+1][j-1]) + (sobelMatY[2][1] * image[i+1][j]) + \
                 (sobelMatY[2][2] * image[i+1][j+1])     
            
            g = int((gx**2 + gy**2)**0.5)
            if(g>255):
                sobelMat[i-1][j-1] = 255
            elif(g<0):
                sobelMat[i-1][j-1] = 0
            else:
                sobelMat[i-1][j-1] = g
    return sobelMat

###############################################################################
img = cv2.imread("hough.jpg", 0)
I = Sobel(img)
cv2.imwrite("tempimage.jpg", I)

###############################################################################
####################  Red lines  ##############################################

dLength = int((img.shape[0]**2 + img.shape[1]**2)**0.5)
pcM = np.zeros((2*dLength+1,11))
thetas = np.arange(-5,5)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(I[i][j]>245):
            polarCoordinates(j, i, dLength, thetas, pcM)

cv2.imwrite("sinH.jpg", pcM)  
 
peakIndex, _ = hough_peaks(pcM, 7)
originalImg = cv2.imread("hough.jpg")
drawLines(originalImg, peakIndex, dLength, thetas)
cv2.imwrite("red_lines.jpg", originalImg)  


###############################################################################
####################  Blue lines  #############################################

dLength = int((img.shape[0]**2 + img.shape[1]**2)**0.5)
pcM = np.zeros((2*dLength+1,5))
thetas = np.arange(-39,-35)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(I[i][j]>160):
            polarCoordinates(j, i, dLength, thetas, pcM)

cv2.imwrite("sinV.jpg", pcM)  
 
peakIndex, aa = hough_peaks(pcM, 13, nSize=17)
originalImg = cv2.imread("hough.jpg")
drawLines(originalImg, peakIndex, dLength, thetas)
cv2.imwrite("blue_lines.jpg", originalImg)


###############################################################################
####################  Coin  ###################################################
def polarCoordinatesCircle(x, y, R1, R2, mat):
    
    for R in range(R1, R2):
        for theta in range(360):
            a = int(x - R*math.cos(math.radians(theta)))
            b = int(y - R*math.sin(math.radians(theta)))
            mat[R-R1][a+R][b+R] += 1

def hough_peaks_circle(mat, nPeaks, nSize=13):
    peakIndex = []
    mat1 = np.copy(mat)
    for i in range(nPeaks):
        iX = np.argmax(mat1)
        R, a, b = np.unravel_index(iX, mat1.shape)         
        peakIndex.append([R, a, b])
        mat1[R][a][b]=0
        
        minR = max(0, R-nSize//2)
        maxR = min(R+nSize//2+1, mat1.shape[0])
        
        minA = max(0, a-nSize//2)
        maxA = min(a+nSize//2+1, mat1.shape[1])
        
        minB = max(0, b-nSize//2)
        maxB = min(b+nSize//2+1, mat1.shape[2])

        for x in range(minR, maxR):
            for y in range(minA, maxA):
                for z in range(minB, maxB):
                    mat1[x][y][z] = 0        
    return peakIndex

def drawCircle(img, pI, R1):
    for i in range(len(pI)):
        R = pI[i][0]+R1
        a = pI[i][1]-R
        b = pI[i][2]-R
        cv2.circle(img, (a, b), R, (255, 255, 0), 2)

R1, R2 = 21, 24
pcMC = np.zeros((R2-R1+1, 2*I.shape[1], 2*I.shape[0]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(I[i][j]>160):
            polarCoordinatesCircle(j, i, R1, R2, pcMC)
            
ind = hough_peaks_circle(pcMC, 17)
originalImg = cv2.imread("hough.jpg")
drawCircle(originalImg, ind, R1)
cv2.imwrite("coin.jpg", originalImg)
