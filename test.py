import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('test.png', 0)  # 以灰度模式讀取

# 二值化
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 過濾 - 中值濾波
denoised_image = cv2.medianBlur(binary_image, 5)
resize_image = cv2.resize(denoised_image,(864,480))
# 顯示原始圖像、二值化後的圖像和過濾後的圖像
cv2.imshow('Denoised Image', resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


