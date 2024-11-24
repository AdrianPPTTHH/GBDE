# -*- coding: utf-8 -*-
"""
Created on Mon October 16 16:48:35 2023

@author: PTH
"""
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import datetime
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance


def division(hb_list, min_synonyms):
    # starttime = datetime.datetime.now()
    
    gb_list_new = []
    k = len(hb_list)
    for hb in hb_list:
        if(len(hb) >= min_synonyms * 2):
           ball_1,ball_2 = spilt_ball(hb)
           DM_parent = get_DM(hb)
           DM_child_1 = get_DM(ball_1)
           DM_child_2 = get_DM(ball_2)
           w = len(ball_1)+len(ball_2)
           w1 = len(ball_1)/w
           w2 = len(ball_2)/w
           w_child = (w1*DM_child_1+w2*DM_child_2)           
           t2 = (w_child < DM_parent)
           if (t2):
               gb_list_new.extend([ball_1,ball_2])
           else:
               gb_list_new.append(hb)
        else:
          gb_list_new.append(hb)

    endtime = datetime.datetime.now()

    return gb_list_new

def spilt_ball(data):
    ball1 = []
    ball2 = []

    caculate_data = data[:,1:]
   
    n, m = caculate_data.shape
    X = caculate_data.T

    # Compute the dot product between each pair of vectors
    G = np.dot(X.T, X)

    # Compute the L2 norms (magnitude) for each vector
    norms = np.array([np.sqrt(np.dot(i, i)) for i in X.T])

    # Compute the outer product of the norms
    H = np.outer(norms, norms)

    # Compute cosine similarity
    cosine_similarity = G / H

    # Convert cosine similarity to cosine distance
    D = 1.0 - cosine_similarity


    # Euclidean distance 
    # n,m = caculate_data.shape
    # X = caculate_data.T
    # G = np.dot(X.T,X)
    # H = np.tile(np.diag(G),(n,1))
    # D = np.sqrt(H + H.T-G*2)
    
    # Minkowski Distance
    # n, m = caculate_data.shape
    # D = np.zeros((n, n))  
    # p = m  
    # for i in range(n):
    #     for j in range(n):
    #         D[i, j] = np.linalg.norm(caculate_data[i, :] - caculate_data[j, :], ord=p)


    r,c = np.where(D == np.max(D))
    
    if len(r) < 2:
        r = np.pad(r, (0, 2 - len(r)), 'constant')
    if len(c) < 2:
        c = np.pad(c, (0, 2 - len(c)), 'constant')

    r1 = r[1]
    c1 = c[1]
    for j in range(0,len(data)):
        if D[j,r1] < D[j,c1]:
            ball1.append(data[j,:])
        else:
            ball2.append(data[j,:])

    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    
    if ball1.shape[0] > 0:
        center1, radius1 = calculate_radius(ball1[:, 1:])
    else:

        # center1 = np.zeros((ball1.shape[1] - 1,))
        radius1 = 0.0

    if ball2.shape[0] > 0:
        center2, radius2 = calculate_radius(ball2[:, 1:])
    else:

        # center2 = np.zeros((ball2.shape[1] - 1,))
        radius2 = 0.0

    
    if(ball1.shape[0] > 0 and ball2.shape[0] > 0):

        distances_to_center1 = [euclidean_distance(x, center1) for x in caculate_data]
        distances_to_center2 = [euclidean_distance(x, center2) for x in caculate_data]

        overlap_mask = (distances_to_center1 <= radius1) & (distances_to_center2 <= radius2)
        overlap_indices = np.where(overlap_mask)[0]
        
        count = 0
        
        ball1_index = [int(x) for x in ball1[:, 0]]
        ball2_index = [int(x) for x in ball2[:, 0]]

        for i in overlap_indices:
            if int(data[i][0]) in ball1_index:
                count += 1
                ball2 = np.append(ball2, [data[i]], axis=0)

            if int(data[i][0]) in ball2_index:
                count += 1
                ball1 = np.append(ball1, [data[i]], axis=0)

    return [ball1, ball2]

def calculate_radius(ball):
    center = np.mean(ball, axis=0)

    radii = [euclidean_distance(x, center) for x in ball]

    radius = np.mean(radii)
    return center, radius

def cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def get_DM(hb):
    if(hb.shape[0] == 0):
        return 0
    
    hb = hb[:,1:]
    num = len(hb)
    center = hb.mean(0)
    
    # distances = np.array([distance.cosine(center, point) for point in hb])
    distances = np.array([cosine_distance(center, point) for point in hb])
    sum_radius = distances.sum()
    radius = distances.max()
    mean_radius = sum_radius / num

    if mean_radius != 0:
        DM = (sum_radius)/(num)
    else:
        DM = radius
    return DM


# Minkowski Distance
# def get_DM(hb):
#     if(hb.shape[0] == 0):
#         return 0
    
#     hb = hb[:,1:]
#     num = len(hb)
#     center = hb.mean(0)
#     p = 1
#     # p = hb.shape[1]

#     diffMat = np.tile(center, (num, 1)) - hb
#     absDiffMat = np.abs(diffMat)
#     distances = np.power(np.sum(np.power(absDiffMat, p), axis=1), 1/p)

#     sum_radius = 0
#     radius = max(distances)
#     for i in distances:
#         sum_radius = sum_radius + i
#     mean_radius = sum_radius/num
#     dimension = len(hb[0])
#     if mean_radius!=0:
#         DM = (sum_radius)/(num)
#     else:
#         DM = radius
#     return DM


def main(data, min_synonyms):
        
    for d in range(1):

        print(data.shape)

        data = np.unique(data, axis=0)

        hb_list_temp = [data]

        row = np.shape(hb_list_temp)[0]
        col = np.shape(hb_list_temp)[1]

        diff = -1
        
        #split
        while 1:
            ball_number_old = len(hb_list_temp)

            hb_list_temp = division(hb_list_temp, min_synonyms)

            ball_number_new = len(hb_list_temp)
            print("ball_old: ", ball_number_old, "\nball_new: ", ball_number_new,"\n")       
        
            if diff == ball_number_new - ball_number_old:
                break
            diff = ball_number_new - ball_number_old

        result = []
        centers = []
        flag = 0

        for i in range(len(hb_list_temp)):

            centers.append(np.insert(np.mean(hb_list_temp[i], axis=0), 0, i , axis=0))
            result.append(np.insert(hb_list_temp[i], 0, i, axis=1))
        return result, centers



