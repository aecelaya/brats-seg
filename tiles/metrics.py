import SimpleITK as sitk
import numpy as np
import pandas as pd
import ants

'''
This script provides a set of useful tools for medical image segmentation.

Current functions:
DiceSITK - Wrapper for SimpleITK Dice coefficient
HausdorffSITK - Wrapper for SimpleITK Hausdorff distance
SurfaceHausdorff - Calculates the surface Hausdorff distance. Can output mean, 95th percentile, or max surface distance
HausdorffDistance - My implementation of the Hausdorff distance. Can output mean, 95th percentile, or max Hausdorff distance

To use, copy this script to the directory that you are working out of and include the following in your script:
from segtools import *

'''

def dice(pred, truth):
    # Read images
    pred = sitk.ReadImage(pred, sitk.sitkUInt8)
    truth = sitk.ReadImage(truth, sitk.sitkUInt8)
    
    # Create instance of SimpleITK overlap measures filter
    overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
    
    # Execute the filter on the two images
    overlapFilter.Execute(pred, truth)
    
    # Get Dice coefficient from filter
    # You can also get JaccardCoefficient or VolumeSimilarity by changing the method
    # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1LabelOverlapMeasuresImageFilter.html
    dice = overlapFilter.GetDiceCoefficient()
    return dice

def hausdorff_sitk(truth, pred, mode = 'max'):
    # Read images
    pred = sitk.ReadImage(pred, sitk.sitkUInt8)
    truth = sitk.ReadImage(truth, sitk.sitkUInt8)
    
    try:
        # Create instance of SimpleITK Hausdorff filter
        hausdorffFilter = sitk.HausdorffDistanceImageFilter()

        # Execute the filter on the two images
        hausdorffFilter.Execute(pred, truth)

        # Get Hausdorff distance from filter
        if mode == 'mean':
            hausdorffDistance = hausdorffFilter.GetAverageHausdorffDistance()
        elif mode == 'max':
            hausdorffDistance = hausdorffFilter.GetHausdorffDistance()
        return hausdorffDistance
    except:
        print('ERROR! Check your Hausdorff distance function!')
        return None
    
def surface_hausdorff(truth, pred, mode):
    '''
    Compute symmetric surface distances, return mean, 95th percentile, or max of distances
    
    Inputs:
    truth: path to ground truth segmentation nifti file
    pred:  path to predicted segmentation nifti file
    mode:  'mean', '95', 'max': mean of distances, 95th percentile of distances, max of distances respectively
    
    Output:
    Surface hausdorff distance: Boundary error between the two predicted images
    
    Usage:
    meanSurfaceHausdorff = SurfaceHausdorff('/path/to/truth.nii.gz', 'path/to/pred.nii.gz', 'mean')
    
    Reference: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    Last modified: 11.25.2020
    '''
    
    # Read in nifti files
    truth = sitk.ReadImage(truth, sitk.sitkUInt8)
    pred = sitk.ReadImage(pred, sitk.sitkUInt8)
    
    truthDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance = False, useImageSpacing=True))
    truthSurface = sitk.LabelContour(truth)

    statsImageFilter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the truth surface by counting all pixels that are 1.
    statsImageFilter.Execute(truthSurface)
    numTruthSurfacePixels = int(statsImageFilter.GetSum()) 

    predDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(pred, squaredDistance = False, useImageSpacing = True))
    predSurface = sitk.LabelContour(pred)
    
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    
    # Distance from prediction surface to ground truth segmentation
    pred2truthDistanceMap = truthDistanceMap * sitk.Cast(predSurface, sitk.sitkFloat32)
    
    # Distance from ground truth surface to predicted segmentation
    truth2predDistanceMap = predDistanceMap * sitk.Cast(truthSurface, sitk.sitkFloat32)
    
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statsImageFilter.Execute(predSurface)
    numPredSurfacePixels = int(statsImageFilter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    pred2truthDistanceMapArray = sitk.GetArrayViewFromImage(pred2truthDistanceMap)
    pred2truthDistances = list(pred2truthDistanceMapArray[pred2truthDistanceMapArray != 0]) 
    pred2truthDistances = pred2truthDistances + list(np.zeros(numPredSurfacePixels - len(pred2truthDistances)))
    
    truth2predDistanceMapArray = sitk.GetArrayViewFromImage(truth2predDistanceMap)
    truth2predDistances = list(truth2predDistanceMapArray[truth2predDistanceMapArray != 0]) 
    truth2predDistances = truth2predDistances + list(np.zeros(numTruthSurfacePixels - len(truth2predDistances)))
    allSurfaceDistances = pred2truthDistances + truth2predDistances
    
    if mode == 'mean':
        hausdorffSurface = np.mean(allSurfaceDistances)
    elif mode == '95':
        hausdorffSurface = np.percentile(allSurfaceDistances, 95)
    elif  mode == 'max':
        hausdorffSurface = np.max(allSurfaceDistances)
    
    return hausdorffSurface

def hausdorff(truth, pred, mode):
    '''
    Compute symmetric Hausdorff distance, return mean, 95th percentile, or max of distances
    
    Inputs:
    truth: path to ground truth segmentation nifti file
    pred:  path to predicted segmentation nifti file
    mode:  'mean', '95', 'max': mean of distances, 95th percentile of distances, max of distances respectively
    
    Output:
    Hausdorff distance between two sets
    
    Usage:
    meanHausdorff = HausdorffDistance('/path/to/truth.nii.gz', 'path/to/pred.nii.gz', 'mean')
    
    Reference: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    Author: Adrian Celaya
    Last modified: 11.25.2020
    '''
    truth = sitk.ReadImage(truth, sitk.sitkUInt8)
    pred = sitk.ReadImage(pred, sitk.sitkUInt8)
    
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(truth)
    num_truth_pixels = int(statistics_image_filter.GetSum()) 
    
    # Get the number of pixels in the prediction surface by counting all pixels that are 1.
    statistics_image_filter.Execute(pred)
    num_pred_pixels = int(statistics_image_filter.GetSum())
    
    truthDTM = sitk.SignedMaurerDistanceMap(truth, squaredDistance=False, useImageSpacing=True)
    predDTM = sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True)

    pred = sitk.Cast(pred, sitk.sitkFloat32)
    truth = sitk.Cast(truth, sitk.sitkFloat32)
    pred2truth_distance_map = truthDTM * pred
    truth2pred_distance_map = predDTM * truth
    
    # Get all non-zero distances and then add zero distances if required.
    pred2truth_distance_map_arr = sitk.GetArrayViewFromImage(pred2truth_distance_map)
    pred2truth_distances = list(pred2truth_distance_map_arr[pred2truth_distance_map_arr > 0]) 
    pred2truth_distances = pred2truth_distances + list(np.zeros(num_pred_pixels - len(pred2truth_distances)))
    
    truth2pred_distance_map_arr = sitk.GetArrayViewFromImage(truth2pred_distance_map)
    truth2pred_distances = list(truth2pred_distance_map_arr[truth2pred_distance_map_arr > 0]) 
    truth2pred_distances = truth2pred_distances + list(np.zeros(num_truth_pixels - len(truth2pred_distances)))
    all_distances = pred2truth_distances + truth2pred_distances
    
    if mode == 'mean':
        hausdorffDistance = np.mean(all_distances)
    elif mode == '95':
        hausdorffDistance = np.percentile(all_distances, 95)
    elif mode == 'max':
        hausdorffDistance = np.max(all_distances)
    
    return hausdorffDistance