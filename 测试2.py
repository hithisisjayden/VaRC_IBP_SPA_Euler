#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:13:23 2025

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
simulation_runs = 1000 # 10000000 一千万是极限了 再算restarting kernal了

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

# # Benchmark VaR 0.99
# a = 4.2171

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

def var_varc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs, bandwidth=1e-4):
    start_time = time.time()
    
    n_repetitions = 10
        
    VaRs, VaRCs, Samples = [], [], []
    for _ in range(n_repetitions):
        epsilon, D, L = generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs)
        VaR = np.percentile(L, alpha * 100)
        LGDs = epsilon * D
        mask = (L >= VaR - bandwidth) & (L <= VaR + bandwidth)
        VaRC = np.mean(LGDs[mask], axis=0)
        Sample = np.sum(mask)
        
        VaRs.append(VaR)
        VaRCs.append(VaRC)
        Samples.append(Sample)
    
    VaRs = np.array(VaRs)
    VaRCs = np.array(VaRCs)
    Samples = np.array(Samples)
        
    VaR_mean, VaR_se = mean_se(VaRs)
    VaRC_mean = np.mean(VaRCs, axis=0)
    VaRC_se = np.array([np.std(VaRCs[:,i]) / np.sqrt(len(VaRCs[:,i])) for i in range(d)])
    Sample_mean = np.mean(Samples)

    end_time = time.time()
    print(f"Time taken for PMC (VaRC_PMC): {end_time - start_time:.2f} seconds")
    
    return VaR_mean, VaR_se, VaRC_mean, VaRC_se, Sample_mean

# VaR_mean, VaR_se, VaRC_mean, VaRC_se, Sample_mean = var_varc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs=100000)

# New Method
def calculate_numerator(d, a, alpha, LGD_a, LGD_b, simulation_runs):
    p = default_probability(d)
    x_threshold = norm.ppf(1-p)
    
    Z_L = np.random.normal(size=simulation_runs) # 10000 simulation_runs
    # Z_D = np.random.normal(size=simulation_runs)
    # eta_L = np.random.normal(size=(simulation_runs, d))
    # eta_D = np.random.normal(size=(simulation_runs, d))
    
    # X = default_driver(Z_D, eta_D)
    # D = (X > x_threshold).astype(int)
    # D = np.ones((simulation_runs,d))
    def L_minus_obligor(d, L, epsilon, D):
        L_minus_i = []
        for i in range(d):
            L_minus_i.append(L - epsilon[:,i] * D[:,i])
        L_minus_i = np.array(L_minus_i).T
        return L_minus_i
    
    # bbP
    def L_minus_cdf(d, a, L_minus_i, epsilon, D):
        
        bbP = []
        for i in range(d):
            bbP.append(np.sum((L_minus_i[:,i] <= (a-epsilon[:,i])) & (D[:,i]==1)) / np.sum((D[:,i]==1)))
        bbP = np.array(bbP)
        return bbP
    
    def calculate_bbP(d, D, L_minus_i, epsilon, simulation_runs):
        
        Z_D = np.random.normal(size=simulation_runs)
        eta_D = np.random.normal(size=(simulation_runs, d))
        X = default_driver(Z_D, eta_D)
        # D 1000,1
        D = (X > x_threshold).astype(int)
        L = np.sum(epsilon * D, axis = 1)
        L_minus_i = L_minus_obligor(d, L, epsilon, D)  
        
        bbP = np.zeros((simulation_runs, d))
        
        for i in range(d):
            for j in range(simulation_runs):
                bbP[j,i] = (np.sum((L_minus_i[:,i] <= (a-epsilon[j,i])) & (D[:,i]==1)) / np.sum((D[:,i]==1)))
        
        return bbP
    
    def conditional_density(d, LGD_a, LGD_b, epsilon, z_L):
        coefficient = np.sqrt(1 - rho**2)
        result = []
        for i in range(d):
            u = beta.cdf(epsilon[:,i], LGD_a, LGD_b)
            v = norm.ppf(u)
            w = (v - rho * z_L) / coefficient
            A = beta.pdf(epsilon[:,i], LGD_a, LGD_b)
            B = norm.pdf(w)
            C = norm.pdf(v)
            result.append((A * B) / (coefficient * C))
        result = np.array(result).T
        return result
    
    def conditional_density_one(LGD_a, LGD_b, epsilon, z_L):
        coefficient = np.sqrt(1 - rho**2)
        u = beta.cdf(epsilon, LGD_a, LGD_b)
        v = norm.ppf(u)
        w = (v - rho * z_L) / coefficient
        A = beta.pdf(epsilon, LGD_a, LGD_b)
        B = norm.pdf(w)
        C = norm.pdf(v)
        return (A * B) / (coefficient * C)
    
    def conditional_density_derivative_inner_expectation(d, LGD_a, LGD_b, epsilon, z_L):
        coefficient = np.sqrt(1 - rho**2)
        result = []
        for i in range(d):
            u = beta.cdf(epsilon[:,i], LGD_a, LGD_b)
            v = norm.ppf(u)
            w = (v - rho * z_L) / coefficient
            A = beta.pdf(epsilon[:,i], LGD_a, LGD_b) / norm.pdf(v)
            B = v - w / coefficient
            C = (LGD_a - 1) / epsilon[:,i] - (LGD_b - 1) / (1 - epsilon[:,i])
            result.append(A * B + C)
        result = np.array(result).T
        return result
    
    inner_expectation = [] # E[ | Z]
    for z_L in Z_L:      
        # z_L_copy = z_L 
        z_L = np.full(simulation_runs, z_L)
        
        Z_D = np.random.normal(size=simulation_runs)
        eta_L = np.random.normal(size=(simulation_runs, d))
        eta_D = np.random.normal(size=(simulation_runs, d))
        
        X = default_driver(Z_D, eta_D)
        D = (X > x_threshold).astype(int)
        
        Y = loss_driver(z_L, eta_L)
        epsilon = beta.ppf(norm.cdf(Y), LGD_a, LGD_b)

        L = np.sum(epsilon * D, axis = 1)
        L_minus_i = L_minus_obligor(d, L, epsilon, D)
        bbP = calculate_bbP(d, D, L_minus_i, epsilon, simulation_runs)

        A = 0
        Term = 1 + epsilon * conditional_density_derivative_inner_expectation(d, LGD_a, LGD_b, epsilon, z_L)
        B_inner = bbP * Term      
        B = np.mean(B_inner, axis = 0)
        inner_expectation.append(A + B) 

    inner_expectation = np.array(inner_expectation)
    return p * np.mean(inner_expectation, axis=0)

def calculate_denominator_Euler(a, numerator):
    # denominator by Euler allocation
    return np.sum(numerator) / a

def varc_IBP_Euler(d, a, alpha, LGD_a, LGD_b, simulation_runs):
    start_time = time.time()
    numerator = calculate_numerator(d, a, alpha, LGD_a, LGD_b, simulation_runs)
    denominator = calculate_denominator_Euler(a, numerator)
    end_time = time.time()
    print(f"Time taken for IBP_Euler (VaRC): {end_time - start_time:.2f} seconds")
    return numerator / denominator
    # return numerator

# VaRCs_Benchmark
# 0.95
VaRCs_pmc = np.array([0.0568,0.1589,0.2315,0.3154,0.3491,0.3906,0.4070,0.4344,0.4341,0.4334])
VaRCs_se_pmc = np.array([0.0034, 0.0062, 0.0062, 0.0048, 0.0052, 0.0045, 0.0046, 0.0041, 0.0032, 0.0037])

# # 0.99
# VaRCs_pmc = np.array([0.1419, 0.2853, 0.3592, 0.4195, 0.4802, 0.4777, 0.5152, 0.5177, 0.5157, 0.5048])
# VaRCs_se_pmc = np.array([0.0117, 0.0132, 0.0114, 0.0086, 0.0057, 0.0088, 0.0057, 0.0089, 0.0084, 0.0075])


# VaR_pmc, VaR_se_pmc, VaRCs_pmc, VaRCs_se_pmc, sample_num_pmc = var_varc_pmc(d, alpha, LGD_a, LGD_b, 10000000)
# VaRCs_IBP_Euler = varc_IBP_Euler(d, a, alpha, LGD_a, LGD_b, simulation_runs)
n_repetitions = 10

VaRCs_IBP_Euler_outputs = np.array([varc_IBP_Euler(d, a, alpha, LGD_a, LGD_b, simulation_runs) for _ in range(n_repetitions)])

VaRCs_IBP_Euler = np.zeros(d)
VaRCs_se_IBP_Euler = np.zeros(d)
for i in range(d):
    VaRCs_IBP_Euler[i], VaRCs_se_IBP_Euler[i] = mean_se(VaRCs_IBP_Euler_outputs[:,i])
    

# Ratio = VaRCs_pmc / VaRCs_IBP_Euler

print(VaRCs_pmc)
print(VaRCs_IBP_Euler)
print(np.sum(VaRCs_IBP_Euler))
# print(Ratio)

# VaRC
plt.figure(figsize=(12, 8), dpi = 300)
x = ['1','2','3','4','5','6','7','8','9','10']

plt.plot(x, VaRCs_pmc, 
            color = 'blue',
            linestyle='-.',
            linewidth=2,
            marker='o',
            markersize=6,
            label='Benchmark')

plt.plot(x, VaRCs_IBP_Euler,  
            color='orange',  
            linestyle='--',
            linewidth=2.5,
            marker='^',
            markersize=6,
            label='IBP_Euler')

plt.title('VaRC 0.95 of Plain MC and IBP_Euler', fontsize=14)
plt.xlabel('Obligors', fontsize=12)
plt.ylabel('VaRC', fontsize=12)
plt.savefig('VaRC 0.95 of Plain MC and IBP_Euler.jpg')

# VaRC_SE
plt.figure(figsize=(12, 8), dpi = 300)
x = ['1','2','3','4','5','6','7','8','9','10']

plt.plot(x, VaRCs_se_pmc, 
            color = 'blue',
            linestyle='-.',
            linewidth=2,
            marker='o',
            markersize=6,
            label='Benchmark')

plt.plot(x, VaRCs_se_IBP_Euler,  
            color='orange',  
            linestyle='--',
            linewidth=2.5,
            marker='^',
            markersize=6,
            label='IBP_Euler')

plt.title('VaRC 0.95 S.E. of Plain MC and IBP_Euler', fontsize=14)
plt.xlabel('Obligors', fontsize=12)
plt.ylabel('VaRC S.E.', fontsize=12)

plt.plot(VaRCs_se_pmc)
plt.plot(VaRCs_se_IBP_Euler)
plt.savefig('VaRC 0.95 S.E. of Plain MC and IBP_Euler.jpg')

# def save_arrays_csv_method2(array1, array2, filename='output.csv'):
#     df = pd.DataFrame({
#         'Array1': array1,
#         'Array2': array2
#     })
#     df.to_csv(filename, index=False)

# save_arrays_csv_method2(VaRCs_pmc, VaRCs_IBP_Euler, 'VaRC_IBP_Euler.csv')


