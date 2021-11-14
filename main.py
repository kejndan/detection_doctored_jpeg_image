import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view
from skimage.filters.rank import median
import time
start = time.time()
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
        kernel_e = np.ones((1,33))
        kernel_mid = (33, 1)
    elif axis == 0:
        kernel_e = np.ones((33,1))
        kernel_mid = (1, 33)
    es_matrix = cv2.filter2D(d, -1, kernel_e)
    show_image(es_matrix)
    return es_matrix - median_filter(es_matrix, kernel_mid)

def extract_blocks(a, blocksize):
    M,N = a.shape
    b0, b1 = blocksize
    return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2).reshape(-1,b0,b1)


AC = 33
img = read_image('planes.jpg').astype(np.float32)
img = np.pad(img, (((AC - 1) // 2, (AC - 1) // 2), ((AC - 1) // 2, (AC - 1) // 2)), mode='reflect')



grad_x, grad_y = calc_diff(img)

# show_image(grad_x)
# show_image(grad_y)
e_2 = calc_e(grad_x, axis=0)
e_1 = calc_e(grad_y, axis=1)
# show_image(e_1)

# show_image(e_2)
#
filter = np.zeros((33, 1))
filter[[0,7, 16, 23, -1]] = 1

e_1_min = e_1.min()
e_1_max = e_1.max()
e_1 = (e_1 - e_1_min)/(e_1_max-e_1_min)

g_h = median(e_1, filter)
g_h = g_h*(e_1_max-e_1_min) + e_1_min

e_2_min = e_2.min()
e_2_max = e_2.max()
e_2 = (e_2 - e_2_min)/(e_2_max-e_2_min)

g_v = median(e_2, filter.T)
g_v = g_v*(e_2_max-e_2_min) + e_2_min
# g_h = np.median(sliding_window_view(e_1, (33, 33))[:,:, [0,7, 16, 23, -1]],axis=(2,3))
# g_v = np.median(sliding_window_view(e_2, (33, 33))[:,:, :,[0,7, 16, 23, -1]],axis=(2,3))

g = g_h + g_v
print(g.mean())
# show_image(g)
g = np.pad(g, ((0, 8 - g.shape[0]%8), (0, 8 - g.shape[1]%8)))
blocks = extract_blocks(g, (8,8))

max_1 = np.max(np.sum(blocks[:, 1:7, 1:7], axis=1), axis=1)
max_2 = np.max(np.sum(blocks[:, 1:7, 1:7], axis=2), axis=1)
min_1 = np.min(np.sum(blocks[:, 1:7, [0,7]], axis=1), axis=1)
min_2 = np.min(np.sum(blocks[:, [0,7], 1:7], axis=2), axis=1)
b = max_1 + max_2 - min_1 - min_2
show_image(b.reshape(g.shape[0]//8,g.shape[1]//8))
# show_image(e_1)
# show_image(e_2)
print(grad_x.min(), grad_y.min())
print(time.time()-start)
# show_image(grad_y)
# show_image(grad_x)
# show_image(img)

