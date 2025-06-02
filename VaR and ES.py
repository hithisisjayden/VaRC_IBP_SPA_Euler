#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:54:17 2025

@author: jaydenwang
"""
import pandas as pd
import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import time
import csv

np.random.seed(0)

# Portfolio size
d = 10

# Repetition for scenarios
n_repetitions = 10

# Simulation paths
simulation_runs = 100000 # 10000000 一千万是极限了 再算restarting kernal了

# Bandwidth
bandwidth = 1e-4

# Confidence level
alpha = 0.95

# LGD Shape parameters
LGD_a, LGD_b = 2, 5

# rho
rho = np.sqrt(0.5)

# Benchmark VaR 0.95
a = 3.2104

# Default probability function
def default_probability(d):
    # k_values = np.arange(1, d + 1)  
    # return 0.01 * (1 + np.sin(16 * k_values * np.pi / d))
    return np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    # return np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

def loss_driver(common_factor, idiosyncratic_factor):
    coefficient = np.sqrt(0.5)
    return coefficient * common_factor[:, np.newaxis] + np.sqrt(1 - coefficient ** 2) * idiosyncratic_factor

def default_driver(common_factor, idiosyncratic_factor):
    coefficient = np.sqrt(0.5)
    return coefficient * common_factor[:, np.newaxis] + np.sqrt(1 - coefficient ** 2) * idiosyncratic_factor

def generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs):
    Z_L = np.random.normal(size=simulation_runs)
    Z_D = np.random.normal(size=simulation_runs)
    eta_L = np.random.normal(size=(simulation_runs, d))
    eta_D = np.random.normal(size=(simulation_runs, d))
    Y = loss_driver(Z_L, eta_L)
    X = default_driver(Z_D, eta_D)
    epsilon = beta.ppf(norm.cdf(Y), LGD_a, LGD_b)
    p = default_probability(d)
    x_threshold = norm.ppf(1-p)
    D = (X > x_threshold).astype(int)
    L = np.sum(epsilon * D, axis = 1)
    return epsilon, D, L

def mean_se(array):
    mean = np.mean(array)
    se = np.std(array) / np.sqrt(len(array))
    return mean, se

def var_varc_es_esc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs, bandwidth=1e-4):
    start_time = time.time()
    
    n_repetitions = 10
        
    VaRs, VaRCs, Samples_varc, ESs, ESCs, Samples_esc= [], [], [], [], [], []
    for _ in range(n_repetitions):
        epsilon, D, L = generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs)
        VaR = np.percentile(L, alpha * 100)
        ES = np.mean(L[L >= VaR])
        
        LGDs = epsilon * D
        mask_varc = (L >= VaR - bandwidth) & (L <= VaR + bandwidth)
        mask_esc = (L >= VaR)
        
        VaRC = np.mean(LGDs[mask_varc], axis=0)
        ESC = np.mean(LGDs[mask_esc], axis=0)
        Sample_varc = np.sum(mask_varc)
        Sample_esc = np.sum(mask_esc)
        
        VaRs.append(VaR)
        VaRCs.append(VaRC)
        ESs.append(ES)
        ESCs.append(ESC)
        Samples_varc.append(Sample_varc)
        Samples_esc.append(Sample_esc)
    
    VaRs = np.array(VaRs)
    VaRCs = np.array(VaRCs)
    ESs = np.array(ESs)
    ESCs = np.array(ESCs)
    Samples_varc = np.array(Samples_varc)
    Samples_esc = np.array(Samples_esc)    
    
    VaR_mean, VaR_se = mean_se(VaRs)
    VaRC_mean = np.mean(VaRCs, axis=0)
    VaRC_se = np.array([np.std(VaRCs[:,i]) / np.sqrt(len(VaRCs[:,i])) for i in range(d)])
    Sample_varc_mean = np.mean(Samples_varc)

    ES_mean, ES_se = mean_se(ESs)
    ESC_mean = np.mean(ESCs, axis=0)
    ESC_se = np.array([np.std(ESCs[:,i]) / np.sqrt(len(ESCs[:,i])) for i in range(d)])
    Sample_esc_mean = np.mean(Samples_esc)

    end_time = time.time()
    print(f"Time taken for PMC (VaRC_PMC): {end_time - start_time:.2f} seconds")
    
    return VaR_mean, VaR_se, VaRC_mean, VaRC_se, Sample_varc_mean, ES_mean, ES_se, ESC_mean, ESC_se, Sample_esc_mean

Result = var_varc_es_esc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs)


















