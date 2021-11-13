import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
def read_image(path):
    rgb_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    return rgb_image

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def calc_diff(img):
    kernel = np.array([[-1,2, -1]])
    Ix = cv2.filter2D(img,-1,kernel)
    Iy = cv2.filter2D(img,-1,kernel.T)
    return np.abs(Ix), np.abs(Iy)


img = read_image('planes.jpg').astype(np.float32)
img_padded = img





def calc_e(d):
    kernel = np.ones((33,1))
    es_matrix = cv2.filter2D(d, -1, kernel)
    return es_matrix

kernel = np.array([-1, 2, 1])
grad_x, grad_y = calc_diff(img_padded)

es_matrix = calc_e(grad_y)

mid_es_matrix = median_filter(es_matrix, (1, 33))

new_es_matrix = es_matrix - mid_es_matrix
show_image(new_es_matrix)
print(grad_x.min(), grad_y.min())
# show_image(grad_y)
# show_image(grad_x)
# show_image(img)