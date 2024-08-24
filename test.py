import numpy as np
import matplotlib.pyplot as plt
def calculate_temperature(T_env, T_0, R, rho, c, V, t):
    # 计算人体温度随时间的变化
    T_body = T_env + (T_0 - T_env) * np.exp(-t / (R * rho * c * V))
    return T_body

# 给定参数

# 体积 (单位: m^3) 暂时假设体积都一样
V1 = 1 

t = np.linspace(0, 5400, 5401)  # 时间范围，0到3600秒 (1小时)，100个点

# 计算人体温度随时间的变化
T_body1 = calculate_temperature(T_env, T_0, R1, rho1, c1, V1, t)
T_body2 = calculate_temperature(T_env, T_0, R2, rho2, c2, V1, t)
T_body3 = calculate_temperature(T_env, T_0, R3, rho3, c3, V1, t)
T_body4 = calculate_temperature(T_env, T_0, R4, rho4, c4, V1, t)


# 输出温度随时间的变化结果
print("时间 (秒)\t人体温度 (K)")
for i in range(len(t)):
    print(f"{t[i]:.1f}\t{T_body[i]:.2f}")

# # 绘制温度随时间变化的曲线
# plt.plot(t, T_body)
# plt.xlabel('时间 (秒)')
# plt.ylabel('人体温度 (K)')
# plt.title('人体温度随时间的变化')
# plt.grid(True)
# plt.show()
