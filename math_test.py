import numpy as np
import matplotlib.pyplot as plt

def point_distance_line(point, line_point1, line_point2):
    if np.array_equal(line_point1, line_point2):
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array - point1_array)

    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]

    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))

    H = A * (A * point[0] + B * point[1] + C) / (A**2 + B**2)
    K = B * (A * point[0] + B * point[1] + C) / (A**2 + B**2)
    h_point = np.array([H, K])

    return distance, h_point

# 測試
point_A = np.array([0, 0])
line_point1_B = np.array([3, 0])
line_point2_C = np.array([0, 4])

distance, h_point = point_distance_line(point_A, line_point1_B, line_point2_C)

# 繪製點A、直線BC和座標點H
plt.plot([line_point1_B[0], line_point2_C[0]], [line_point1_B[1], line_point2_C[1]], label='Line BC')
plt.scatter(point_A[0], point_A[1], color='red', label='Point A')
plt.scatter(h_point[0], h_point[1], color='green', label='Point H (Perpendicular)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
