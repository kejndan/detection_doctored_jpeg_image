import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view
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

def calc_e(d, axis=0):
    if axis == 1:
        kernel_e = kernel.T
        kernel_mid = (33, 1)
    elif axis == 0:
        kernel_e = np.ones((33,1))
        kernel_mid = (1, 33)
    es_matrix = cv2.filter2D(d, -1, kernel_e)
    return es_matrix - median_filter(es_matrix, kernel_mid)


img = read_image('planes.jpg').astype(np.float32)
img_padded = img


kernel = np.array([-1, 2, 1])
grad_x, grad_y = calc_diff(img_padded)

e_1 = calc_e(grad_y)
e_2 = calc_e(grad_x, 1)

g_h = np.median(sliding_window_view(e_1, (33, 33))[:,:, [0,7, 16, 23, -1]],axis=(2,3))
g_v = np.median(sliding_window_view(e_1, (33, 33))[:,:, :,[0,7, 16, 23, -1]],axis=(2,3))

g = g_h + g_v
show_image(g)

show_image(e_1)
show_image(e_2)
print(grad_x.min(), grad_y.min())
# show_image(grad_y)
# show_image(grad_x)
# show_image(img)