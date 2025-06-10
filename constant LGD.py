#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:40:23 2025

@author: jaydenwang
"""

import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import root_scalar
from numpy.polynomial.hermite import hermgauss
from functools import lru_cache

# --- 1. 参数设置 ---
# 根据您的文档和要求进行设置
d = 10  # 投资组合规模 (portfolio size)
p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]) # 无条件违约概率
epsilon = np.ones(d)  # 固定LGD，这里假设为1
rho = np.full(d, np.sqrt(0.5))  # 相关性因子 rho_i = sqrt(0.5) 
alpha = 0.97  # 置信水平
# VaR(a)的值，来自您文档中的表格 Table 1, VaR_0.97 = 3.5507 
a_VaR = 4
n_quad_points = 32  # 高斯-埃尔米特求积点数
repetitions = 5 # 重复计算次数

# 计算正态分布的阈值 x_i^d
x_d = norm.ppf(p)

# --- 2. 辅助函数 ---
def p_i_zD(z, p_i_uncond, rho_i):
    """
    计算给定共同因子 zD 下的条件违约概率 p_i(zD)。
    公式来源: Page 4 
    """
    # 调整 x_d 的计算以匹配公式 P(D_i=1)=p_i
    # P(X_i > x_i^d) = P(rho*Z + sqrt(1-rho^2)*eta > x_i^d) = p_i
    # X_i|Z=z ~ N(rho*z, 1-rho^2)
    # P(X_i > x_i^d | Z=z) = 1 - Phi((x_i^d - rho*z)/sqrt(1-rho^2))
    # 文档中 x_i^d = Phi^-1(1-p_i), 但这会导致 p_i(z) = Phi((rho*z - Phi^-1(1-p_i))/sqrt(1-rho^2))
    # 让我们遵循文档 Page 4 的公式：p_i(zD) = 1 - Phi((x_i^d - rho_i*z)/sqrt(1-rho_i^2))
    # 其中 x_i^d = Phi^-1(p_i) 似乎与 D_i = I(X_i >= x_i^d) 定义更一致, 而非文档中的 Phi^-1(1-p_i).
    # 如果 D_i = I(X_i > x_i^d) 且 P(D_i=1)=p_i, 那么 P(X_i > x_i^d)=p_i, 所以 x_i^d = Phi^-1(1-p_i)
    x_i_d = norm.ppf(1 - p_i_uncond)
    return 1 - norm.cdf((x_i_d - rho_i * z) / np.sqrt(1 - rho_i**2))


# --- 3. 分子计算模块 (Numerator Calculation) ---
# 该模块计算 f_{L_{-i}|D_i=1}(a - e_i)

@lru_cache(maxsize=None)
def C_m(t, z, j_excluded, m):
    """
    计算 C^(m)(t, z^D)，即 log(Q) 的 m 阶导数。
    公式来源: Page 5 
    """
    p_j_z = np.array([p_i_zD(z, p[j], rho[j]) for j in range(d) if j != j_excluded])
    e_j = np.array([epsilon[j] for j in range(d) if j != j_excluded])
    
    numerator = p_j_z * (e_j**m) * np.exp(t * e_j)
    denominator = 1 - p_j_z + p_j_z * np.exp(t * e_j)
    
    # 防止分母为0
    denominator[denominator == 0] = 1e-100
    
    return np.sum(numerator / denominator)

@lru_cache(maxsize=None)
def Q_k(t, z, j_excluded, k):
    """
    使用递推公式计算 Q^(k)(t, z^D)。
    公式来源: Page 5 
    """
    if k == 0:
        p_j_z = np.array([p_i_zD(z, p[j], rho[j]) for j in range(d) if j != j_excluded])
        e_j = np.array([epsilon[j] for j in range(d) if j != j_excluded])
        return np.prod(1 - p_j_z + p_j_z * np.exp(t * e_j))
    
    res = 0
    for q in range(k):
        # 使用 scipy.special.comb 计算二项式系数
        from scipy.special import comb
        res += comb(k - 1, q) * Q_k(t, z, j_excluded, q) * C_m(t, z, j_excluded, k - q)
    return res

def I_k(t, j_excluded, k, n_points):
    """
    使用高斯-埃尔米特求积计算 I^(k)(t)。
    公式来源: Page 5-6 
    """
    nodes, weights = hermgauss(n_points)
    
    # 变量代换 z = sqrt(2) * x
    z_nodes = np.sqrt(2) * nodes
    
    integrand = np.array([
        p_i_zD(z, p[j_excluded], rho[j_excluded]) * Q_k(t, z, j_excluded, k)
        for z in z_nodes
    ])
    
    integral = np.dot(weights, integrand) / np.sqrt(np.pi)
    return integral

def K_L_minus_i_derivatives(t, j_excluded, n_points):
    """
    计算 K_{L_{-i}|D_i=1}(t) 的前四阶导数。
    公式来源: Page 5 
    """
    I0 = I_k(t, j_excluded, 0, n_points)
    I1 = I_k(t, j_excluded, 1, n_points)
    I2 = I_k(t, j_excluded, 2, n_points)
    I3 = I_k(t, j_excluded, 3, n_points)
    I4 = I_k(t, j_excluded, 4, n_points)

    K1 = I1 / I0
    K2 = (I2 * I0 - I1**2) / I0**2
    K3 = (I3 * I0**2 - 3 * I2 * I1 * I0 + 2 * I1**3) / I0**3
    K4 = (I4 * I0**3 - 4*I3*I1*I0**2 - 3*I2**2*I0**2 + 12*I2*I1**2*I0 - 6*I1**4) / I0**4
    
    return K1, K2, K3, K4

def solve_saddlepoint_t_numerator(x, j_excluded, n_points):
    """解鞍点方程 K'(t) = x"""
    def equation(t):
        k1, _, _, _ = K_L_minus_i_derivatives(t, j_excluded, n_points)
        return k1 - x
    # 使用 root_scalar 寻找根
    try:
        sol = root_scalar(equation, bracket=[-10, 10], method='brentq')
        return sol.root
    except ValueError:
        return np.nan # 如果在区间内找不到根

def pdf_L_minus_i(x, j_excluded, n_points):
    """
    使用二阶SPA计算 f_{L_{-i}|D_i=1}(x)。
    公式来源: Page 6 
    """
    t_hat = solve_saddlepoint_t_numerator(x, j_excluded, n_points)
    if np.isnan(t_hat):
        return 0.0

    # 计算在 t_hat 处的c.g.f.值和导数
    p_i_uncond = p[j_excluded]
    I0_at_t_hat = I_k(t_hat, j_excluded, 0, n_points)
    K0 = -np.log(p_i_uncond) + np.log(I0_at_t_hat)
    
    K1, K2, K3, K4 = K_L_minus_i_derivatives(t_hat, j_excluded, n_points)

    if K2 <= 0: return 0.0 # 无效的二阶导
    
    lambda3 = K3 / (K2**1.5)
    lambda4 = K4 / (K2**2.0)
    
    term_in_braces = 1 + (1/8) * lambda4 - (5/24) * lambda3**2
    
    pdf = np.exp(K0 - t_hat * x) / np.sqrt(2 * np.pi * K2) * term_in_braces
    return pdf


# --- 4. 分母计算模块 (Denominator Calculation) ---
# 该模块计算 f_L(a)

def K_L_derivatives_zD(t, z):
    """
    计算给定 zD 下 L 的c.g.f. K(t; zD) 的前四阶导数。
    公式来源: Page 6 
    """
    p_z = np.array([p_i_zD(z, p_i, rho_i) for p_i, rho_i in zip(p, rho)])
    e = epsilon
    
    term = 1 - p_z + p_z * np.exp(t * e)
    term[term == 0] = 1e-100 # 防止除以0
    
    K1 = np.sum(p_z * e * np.exp(t * e) / term)
    K2 = np.sum((p_z * (e**2) * np.exp(t * e) * (1 - p_z)) / term**2)
    
    # 简化高阶导数计算
    d1 = p_z * e * np.exp(t*e)
    d2 = p_z * e**2 * np.exp(t*e)
    d3 = p_z * e**3 * np.exp(t*e)
    d4 = p_z * e**4 * np.exp(t*e)

    A = 1 - p_z
    B = p_z * np.exp(t*e)
    S = A + B

    K3 = np.sum(d3/S - 3*d2*d1/S**2 + 2*d1**3/S**3)
    K4 = np.sum(d4/S - 4*d3*d1/S**2 - 3*d2**2/S**2 + 12*d2*d1**2/S**3 - 6*d1**4/S**4)

    return K1, K2, K3, K4

def solve_saddlepoint_t_denominator(x, z):
    """解鞍点方程 K'(t; zD) = x"""
    def equation(t):
        k1, _, _, _ = K_L_derivatives_zD(t, z)
        return k1 - x
    try:
        sol = root_scalar(equation, bracket=[-10, 10], method='brentq')
        return sol.root
    except ValueError:
        return np.nan

def pdf_L_zD(x, z):
    """
    使用二阶SPA计算条件PDF f_L(x|zD)。
    公式来源: Page 6 
    """
    t_hat = solve_saddlepoint_t_denominator(x, z)
    if np.isnan(t_hat):
        return 0.0

    p_z = np.array([p_i_zD(z, pi, rho_i) for pi, rho_i in zip(p, rho)])
    K0 = np.sum(np.log(1 - p_z + p_z * np.exp(t_hat * epsilon)))
    
    K1, K2, K3, K4 = K_L_derivatives_zD(t_hat, z)
    
    if K2 <= 0: return 0.0
    
    lambda3 = K3 / (K2**1.5)
    lambda4 = K4 / (K2**2.0)
    
    term_in_braces = 1 + (1/8) * lambda4 - (5/24) * lambda3**2
    
    pdf = np.exp(K0 - t_hat * x) / np.sqrt(2 * np.pi * K2) * term_in_braces
    return pdf

def pdf_L(x, n_points):
    """
    通过对zD积分计算无条件PDF f_L(x)。
    """
    nodes, weights = hermgauss(n_points)
    z_nodes = np.sqrt(2) * nodes
    
    integrand = np.array([pdf_L_zD(x, z) for z in z_nodes])
    
    integral = np.dot(weights, integrand) / np.sqrt(np.pi)
    return integral

# --- 5. 主计算流程 ---
if __name__ == "__main__":
    print("开始计算 VaR Contributions (Constant LGD Case)...")
    print(f"参数设置: d={d}, alpha={alpha}, VaR(a)={a_VaR}, Repetitions={repetitions}")
    
    start_time = time.time()
    
    all_varc_results = []

    for rep in range(repetitions):
        print(f"\n--- 第 {rep + 1}/{repetitions} 次计算 ---")
        
        # 清除缓存以确保每次重复计算的独立性
        C_m.cache_clear()
        Q_k.cache_clear()

        # 1. 计算分母 f_L(a)
        denominator_f_L_a = pdf_L(a_VaR, n_quad_points)
        print(f"分母 f_L(a={a_VaR:.4f}) = {denominator_f_L_a:.6f}")

        varc_i_results = []
        for i in range(d):
            # 2. 计算分子中的 f_{L_{-i}|D_i=1}(a - e_i)
            x_for_pdf = a_VaR - epsilon[i]
            numerator_pdf = pdf_L_minus_i(x_for_pdf, i, n_quad_points)
            
            # 3. 计算VaRC_i
            numerator = p[i] * epsilon[i] * numerator_pdf
            if denominator_f_L_a > 0:
                varc_i = numerator / denominator_f_L_a
            else:
                varc_i = np.nan
            
            varc_i_results.append(varc_i)
            print(f"  Obligor {i+1}: VaRC = {varc_i:.6f}")
            
        all_varc_results.append(varc_i_results)

    end_time = time.time()
    
    # --- 6. 结果汇总与输出 ---
    all_varc_results = np.array(all_varc_results)
    mean_varc = np.nanmean(all_varc_results, axis=0)
    std_err_varc = np.nanstd(all_varc_results, axis=0) / np.sqrt(repetitions)
    
    print("\n" + "="*50)
    print("计算完成: 最终结果汇总")
    print("="*50)
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每次耗时: {(end_time - start_time) / repetitions:.2f} 秒")
    print("\n--- VaRC 均值与标准误 ---")
    print("Obligor |   VaRC (Mean)   |   Std. Error")
    print("--------|-----------------|---------------")
    for i in range(d):
        print(f"  {i+1:<5} |    {mean_varc[i]:.8f}   |   {std_err_varc[i]:.8f}")
    print("\n--- VaRC 求和 ---")
    print(f"所有VaRC贡献之和: {np.sum(mean_varc):.6f}")
    print(f"理论VaR(a): {a_VaR:.6f}")
    print(f"相对误差: {abs(np.sum(mean_varc) - a_VaR) / a_VaR * 100:.4f}%")