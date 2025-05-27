#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:13:23 2025

@author: jaydenwang
"""

import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import time

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

# def var_varc_pmc(d, alpha, LGD_a, LGD_b, simulation_runs, bandwidth=1e-4):
    
    start_time_pmc = time.time()
    
    Results = [generate_samples_pmc(d, alpha, LGD_a, LGD_b, simulation_runs) for _ in range(n_repetitions)]
    # end_time_pmc = time.time()
    epsilon, D, L = zip(*Results) # type: tuple
    VaRs = np.array([np.percentile(i, alpha * 100) for i in L])
    VaR, VaR_se = mean_se(VaRs)   
    
    end_time_pmc = time.time()
    print(f"Time taken for PMC (VaR_PMC): {end_time_pmc - start_time_pmc:.2f} seconds")
    
    start_time_varc_calc = time.time()

    LGDs_per_repetition = tuple(eps_rep * D_rep for eps_rep, D_rep in zip(epsilon, D))
    
    VaRCs_list_per_rep = []
    sample_num = []
    for k in range(n_repetitions):
        L_k, Li_k, a_k = L[k], LGDs_per_repetition[k], VaRs[k]
        
        mask_k = (L_k >= a_k - bandwidth) & (L_k <= a_k + bandwidth)
        sample_num.append(np.sum(mask_k))
        conditional_Li_k = Li_k[mask_k]
        if conditional_Li_k.shape[0] > 0:
            VaRCs_list_per_rep.append(np.mean(conditional_Li_k, axis=0))
        else:
            VaRCs_list_per_rep.append(np.full(d, np.nan))
            
    VaRCs_matrix = np.array(VaRCs_list_per_rep) # Shape: (n_repetitions, d)

    # Initialize arrays for final VaRC means and SEs
    VaRC_final_means = np.full(d, np.nan)
    VaRC_final_ses = np.full(d, np.nan)

    # Calculate mean and SE for each obligor's VaRC estimates using your mean_se function
    for i in range(d): # For each obligor (column i)
        varc_estimates_for_obligor_i = VaRCs_matrix[:, i]
        valid_estimates = varc_estimates_for_obligor_i[~np.isnan(varc_estimates_for_obligor_i)]
        
        if len(valid_estimates) > 0:
            # Apply your mean_se function to the 1D array of valid estimates for this obligor
            mean_val, se_val = mean_se(valid_estimates)
            VaRC_final_means[i] = mean_val
            VaRC_final_ses[i] = se_val
            
    VaRC = VaRC_final_means
    VaRC_se = VaRC_final_ses
    sample_num = np.mean(np.array(sample_num))
    # end_time_varc_calc = time.time()
    # print(f"Time taken for PMC (VaRC_PMC): {end_time_varc_calc - start_time_varc_calc:.2f} seconds")
    
    return VaR, VaR_se, VaRC, VaRC_se, sample_num

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
    Z_D = np.random.normal(size=simulation_runs)
    eta_L = np.random.normal(size=(simulation_runs, d))
    eta_D = np.random.normal(size=(simulation_runs, d))
    
    def L_minus_obligor(d, L, epsilon):
        L_minus_i = []
        for i in range(d):
            L_minus_i.append(L - epsilon[:,i] * D[:,i])
        L_minus_i = np.array(L_minus_i).T
        return L_minus_i
    
    def L_minus_cdf(d, a, L_minus_i, epsilon, D):
        bbP = []
        for i in range(d):
            bbP.append(np.sum((L_minus_i[:,i] <= (a-epsilon[:,i])) & (D[:,i]==1)) / np.sum((D[:,i]==1)))
        bbP = np.array(bbP)
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
        z_L_copy = z_L
        z_L = np.full(simulation_runs, z_L)
        Y = loss_driver(z_L, eta_L)
        X = default_driver(Z_D, eta_D)
        epsilon = beta.ppf(norm.cdf(Y), LGD_a, LGD_b)
        # epsilon = np.clip(epsilon, 1e-6, 1 - 1e-6)
        D = (X > x_threshold).astype(int)
        L = np.sum(epsilon * D, axis = 1)
        L_minus_i = L_minus_obligor(d, L, epsilon)
        bbP = L_minus_cdf(d, a, L_minus_i, epsilon, D)
        bbP_1 = L_minus_cdf(d, a, L_minus_i, np.full((simulation_runs,d),0.999), D) # e_i cannot be 1
        # bbP = bbP_1 = 0.95
        if a > 1:
            A = - bbP_1 * 0.999 * conditional_density_one(LGD_a, LGD_b, 0.999, z_L_copy)
        else:
            A = - L_minus_cdf(d, a, L_minus_i, np.full((simulation_runs,d),a), D) * np.full((simulation_runs,d),a) * conditional_density(d, LGD_a, LGD_b, np.full((simulation_runs,d),a), z_L_copy)
        A = 0
        B_inner = bbP * (1 + epsilon * conditional_density_derivative_inner_expectation(d, LGD_a, LGD_b, epsilon, z_L))
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
# VaRCs_pmc = np.array([0.0568,0.1589,0.2315,0.3154,0.3491,0.3906,0.4070,0.4344,0.4341,0.4334])

VaR_pmc, VaR_se_pmc, VaRCs_pmc, VaRCs_se_pmc, sample_num_pmc = var_varc_pmc(d, alpha, LGD_a, LGD_b, 10000000)
VaRCs_IBP_Euler = varc_IBP_Euler(d, VaR_pmc, alpha, LGD_a, LGD_b, simulation_runs)
Ratio = VaRCs_pmc / VaRCs_IBP_Euler
print(VaRCs_pmc)
print(VaRCs_IBP_Euler)
print(np.sum(VaRCs_IBP_Euler))
print(Ratio)

plt.plot(np.array([0.0568,0.1589,0.2315,0.3154,0.3491,0.3906,0.4070,0.4344,0.4341,0.4334]))
plt.plot(VaRCs_pmc)
plt.plot(VaRCs_IBP_Euler)



