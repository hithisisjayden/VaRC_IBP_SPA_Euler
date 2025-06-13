import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import root_scalar
from numpy.polynomial.hermite import hermgauss
from functools import lru_cache
from scipy.special import comb

# --- 1. 参数设置 ---
d = 10  # 投资组合规模
p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]) # 无条件违约概率
epsilon = np.ones(d)  # 固定LGD，假设为1
# epsilon = np.array([10,9,8,7,6,5,4,3,2,1])
rho = np.full(d, np.sqrt(0.5))  # 相关性因子
alpha = 0.97  # 置信水平
a_VaR = 4 # VaR(a)的值, 来自您文档Table 1, alpha=0.97 
n_quad_points = 32  # 高斯-埃尔米特求积点数
repetitions = 1 # 算法是确定性的，一次即可

# --- 2. 辅助函数 ---
def p_i_zD(z, p_i_uncond, rho_i):
    """
    计算给定共同因子 zD 下的条件违约概率 p_i(zD)。
    公式来源: Page 4 
    """
    x_i_d = norm.ppf(1 - p_i_uncond)
    return 1 - norm.cdf((x_i_d - rho_i * z) / np.sqrt(1 - rho_i**2))

# --- 3. 分子计算模块 (已修正) ---
@lru_cache(maxsize=None)
def C_m_numerator(t, z, j_excluded, m):
    """计算分子 C^(m)(t, z^D)。公式来源: Page 5 """
    indices = [j for j in range(d) if j != j_excluded]
    p_j_z = np.array([p_i_zD(z, p[j], rho[j]) for j in indices])
    e_j = epsilon[indices]
    
    numerator_term = p_j_z * (e_j**m) * np.exp(t * e_j)
    denominator_term = 1 - p_j_z + p_j_z * np.exp(t * e_j)
    denominator_term[denominator_term == 0] = 1e-120
    return np.sum(numerator_term / denominator_term)

@lru_cache(maxsize=None)
def Q_k_numerator(t, z, j_excluded, k):
    """使用递推公式计算分子 Q^(k)(t, z^D)。公式来源: Page 5 """
    if k == 0:
        indices = [j for j in range(d) if j != j_excluded]
        p_j_z = np.array([p_i_zD(z, p[j], rho[j]) for j in indices])
        e_j = epsilon[indices]
        return np.prod(1 - p_j_z + p_j_z * np.exp(t * e_j))
    
    res = sum(comb(k - 1, q) * Q_k_numerator(t, z, j_excluded, q) * C_m_numerator(t, z, j_excluded, k - q) for q in range(k))
    return res

def I_k_numerator(t, j_excluded, k, n_points):
    """使用高斯-埃尔米特求积计算分子 I^(k)(t)。公式来源: Page 5-6 """
    nodes, weights = hermgauss(n_points)
    z_nodes = np.sqrt(2) * nodes
    
    integrand = np.array([p_i_zD(z, p[j_excluded], rho[j_excluded]) * Q_k_numerator(t, z, j_excluded, k) for z in z_nodes])
    return np.dot(weights, integrand) / np.sqrt(np.pi)

def K_L_minus_i_derivatives(t, j_excluded, n_points):
    """计算 K_{L_{-i}|D_i=1}(t) 的前四阶导数(使用修正后公式)"""
    I0 = I_k_numerator(t, j_excluded, 0, n_points)
    I1 = I_k_numerator(t, j_excluded, 1, n_points)
    I2 = I_k_numerator(t, j_excluded, 2, n_points)
    I3 = I_k_numerator(t, j_excluded, 3, n_points)
    I4 = I_k_numerator(t, j_excluded, 4, n_points)

    if abs(I0) < 1e-120: return np.nan, np.nan, np.nan, np.nan

    K1 = I1 / I0
    K2 = I2 / I0 - (I1 / I0)**2
    K3 = I3 / I0 - 3 * I2 * I1 / I0**2 + 2 * I1**3 / I0**3
    K4 = I4 / I0 - (4 * I3 * I1 + 3 * I2**2) / I0**2 + 12 * I2 * I1**2 / I0**3 - 6 * I1**4 / I0**4

    return K1, K2, K3, K4

def solve_saddlepoint_t(x, derivative_func):
    """通用鞍点方程求解器"""
    def equation(t):
        k1, _, _, _ = derivative_func(t)
        return k1 - x if not np.isnan(k1) else 1e6
    try:
        sol = root_scalar(equation, bracket=[-20, 20], method='brentq')
        return sol.root
    except (ValueError, RuntimeError):
        return np.nan

def pdf_spa(x, t_hat, K0, K2, K3, K4):
    """通用二阶SPA PDF计算器。公式来源: Page 6 """
    if np.isnan(t_hat) or np.isnan(K2) or K2 <= 1e-12: return 0.0
    
    lambda3_sq = (K3**2) / (K2**3)
    lambda4 = K4 / (K2**2)
    
    correction = 1 + (1/8) * lambda4 - (5/24) * lambda3_sq
    
    pdf = np.exp(K0 - t_hat * x) / np.sqrt(2 * np.pi * K2) * correction
    return pdf if pdf > 0 and not np.isnan(pdf) else 0.0

# --- 4. 分母计算模块 (已优化) ---
# def K_L_derivatives_zD(t, z):
#     """计算分母 K(t;zD) 的各阶导数(系统化方法)"""
#     p_z = np.array([p_i_zD(z, pi, rho_i) for pi, rho_i in zip(p, rho)])
#     e = epsilon
#     exp_te = np.exp(t * e)
    
#     # A_i(t) = 1 - p_i(z) + p_i(z) * exp(t*e_i)
#     # K(t) = sum(log(A_i(t)))
#     A = 1 - p_z + p_z * exp_te
#     A_d1 = p_z * e * exp_te
#     A_d2 = p_z * e**2 * exp_te
#     A_d3 = p_z * e**3 * exp_te
#     A_d4 = p_z * e**4 * exp_te

#     # k_i' = A_i'/A_i, k_i'' = ..., K' = sum(k_i'), K'' = sum(k_i'')
#     k1 = A_d1 / A
#     k2 = A_d2/A - k1**2
#     k3 = A_d3/A - 3*A_d2*k1/A - k1**3 + 3*k1*k2
#     k4 = (A_d4/A - 4*A_d3*k1/A - 3*A_d2*k2/A - 6*A_d2*k1**2/A 
#           + 6*A_d1*k1*k2/A + 3*A_d1*k1**3/A) # This is getting complicated. Let's use the explicit formulas from the previous debugged version which are derived from a systematic approach.
#     # Re-deriving k_i'''' from k_i'''
#     k3_prime = (A_d4/A - A_d3*A_d1/A**2) - 3*(A_d3*k1/A + A_d2*k1/A - A_d2*k1*A_d1/A**2) - 3*k1**2*k2 + 3*k2**2 + 3*k1*k3
#     # This is also error prone. The best way is to use the formula K'''' = sum(k_i'''').
#     k1 = A_d1 / A
#     k2 = A_d2 / A - k1**2
#     k3 = (A_d3/A) - 3*(A_d2/A)*k1 - k1**3
#     k4 = (A_d4/A) - 4*(A_d3/A)*k1 - 3*(A_d2/A)**2 + 12*(A_d2/A)*k1**2 - 6*k1**4
    
#     K1 = np.sum(A_d1 / A)
#     K2 = np.sum(A_d2/A - (A_d1/A)**2)
#     K3 = np.sum(A_d3/A - 3*(A_d2/A)*(A_d1/A) + 2*(A_d1/A)**3)
#     K4 = np.sum(A_d4/A - 4*(A_d3/A)*(A_d1/A) - 3*(A_d2/A)**2 + 12*(A_d2/A)*(A_d1/A)**2 - 6*(A_d1/A)**4)
    
#     return K1, K2, K3, K4

# --- 4. 分母计算模块 (已优化) ---
def K_L_derivatives_zD(t, z):
    """
    计算给定 zD 下 L 的c.g.f. K(t; zD) 的前四阶导数。
    公式来源: Page 6 
    """
    p_z = np.array([p_i_zD(z, p_i, rho_i) for p_i, rho_i in zip(p, rho)])
    e = epsilon
    
    A = 1 - p_z
    B = p_z * np.exp(t*e)
    
    K1 = np.sum(e * B / (A + B))
    K2 = np.sum(A * (e**2) * B / (A + B)**2)
    K3 = np.sum(A * (e**3) * B / (A + B)**2 - 2 * A * (e**3) * (B**2) / (A + B)**3)
    K4 = np.sum(A * (e**4) * B / (A + B)**2 - 6 * A * (e**4) * (B**2) / (A + B)**3 + 6 * A * (e**4) * (B**3) / (A + B)**4)    


    return K1, K2, K3, K4

# Euler allocation 计算分母,在一个非常确定非常准确的VaR = a的基础上
def Denominator_Euler(numerators, a):
    return np.sum(numerators) / a


# --- 5. 主计算流程 ---
if __name__ == "__main__":
    print("开始计算 VaR Contributions (使用修正后公式)...")
    start_time = time.time()

    # 1. 计算分母 f_L(a)
    def den_deriv_func(t):
        nodes, weights = hermgauss(n_quad_points)
        z_nodes = np.sqrt(2) * nodes
        # 需要计算 K_L(t) 的导数, K_L(t) = E[K_L(t;z)]
        # K_L'(t) = E[K_L'(t;z)]
        k_ders_at_nodes = [K_L_derivatives_zD(t, z) for z in z_nodes]
        k_ders_avg = np.mean(k_ders_at_nodes, axis=0)
        return k_ders_avg
    
    # We need the PDF f_L(a) = E[f_L(a;z)]
    nodes, weights = hermgauss(n_quad_points)
    z_nodes = np.sqrt(2) * nodes
    
    pdf_L_z_values = []
    for z in z_nodes:
        t_hat_den = solve_saddlepoint_t(a_VaR, lambda t: K_L_derivatives_zD(t, z))
        if np.isnan(t_hat_den):
            pdf_L_z_values.append(0.0)
            continue
        p_z = np.array([p_i_zD(z, pi, rho_i) for pi, rho_i in zip(p, rho)])
        K0_den = np.sum(np.log(1 - p_z + p_z * np.exp(t_hat_den * epsilon)))
        ders_den = K_L_derivatives_zD(t_hat_den, z)
        pdf_L_z_values.append(pdf_spa(a_VaR, t_hat_den, K0_den, *ders_den[1:]))

    denominator_f_L_a = np.dot(weights, pdf_L_z_values) / np.sqrt(np.pi)
    print(f"分母 f_L(a={a_VaR:.4f}) = {denominator_f_L_a:.6f}")
    
    varc_results, numerator_results = [], []
    for i in range(d):
        # 2. 计算分子
        x_for_pdf = a_VaR - epsilon[i]
        
        # 为当前 obligor i 求解鞍点
        t_hat_num = solve_saddlepoint_t(x_for_pdf, lambda t: K_L_minus_i_derivatives(t, i, n_quad_points))
        
        # 计算分子 PDF
        if np.isnan(t_hat_num):
            numerator_pdf = 0.0
        else:
            I0_at_t_hat = I_k_numerator(t_hat_num, i, 0, n_quad_points)
            K0_num = -np.log(p[i]) + np.log(I0_at_t_hat)
            ders_num = K_L_minus_i_derivatives(t_hat_num, i, n_quad_points)
            numerator_pdf = pdf_spa(x_for_pdf, t_hat_num, K0_num, *ders_num[1:])
        
        # 3. 计算VaRC_i
        numerator = p[i] * epsilon[i] * numerator_pdf
        numerator_results.append(numerator)
        varc_i = numerator / denominator_f_L_a if denominator_f_L_a > 0 else 0.0
        varc_results.append(varc_i)

    end_time = time.time()

    # --- 6. 结果汇总与输出 ---
    print("\n" + "="*60)
    print("                 计算完成: 最终结果汇总")
    print("="*60)
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("\n--- VaRC 结果 ---")
    print("Obligor |   VaRC (Mean)   |   p_i")
    print("--------|-----------------|--------")
    for i in range(d):
        print(f"  {i+1:<5} |    {varc_results[i]:.8f}   |  {p[i]:.2f}")
    
    print("\n--- VaRC 求和验证 (Euler Allocation Principle) ---")
    sum_of_varc = np.sum(varc_results)
    print(f"所有VaRC贡献之和: {sum_of_varc:.6f}")
    print(f"理论VaR(a)值:    {a_VaR:.6f}")
    print(f"绝对误差:         {abs(sum_of_varc - a_VaR):.6f}")
    print(f"相对误差:         {abs(sum_of_varc - a_VaR) / a_VaR * 100:.4f}%")