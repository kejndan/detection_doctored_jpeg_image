import numpy as np
from cv2 import cv2
from scipy.ndimage import median_filter
from utils import split_into_blocks


class DetectionDoctored:
    def __init__(self, img):
        self.img = img.astype(np.float32)

    def calc_approx_second_derivative(self):
        kernel = np.array([[-1,2, -1]])
        Ix = cv2.filter2D(self.img, -1, kernel)
        Iy = cv2.filter2D(self.img, -1, kernel.T)
        return np.abs(Ix), np.abs(Iy)

    def first_stage_BAG(self, d, ax):
        if ax == 'horizontal':
            kernel = np.ones((1, 33))
            kernel_mid = (33, 1)
        elif ax == 'vertical':
            kernel = np.ones((33, 1))
            kernel_mid = (1, 33)
        e_matrix = cv2.filter2D(d, -1, kernel)
        return e_matrix - median_filter(e_matrix, kernel_mid)

    def second_stage_BAG(self, e_matrix, ax):
        mask = np.zeros((33, 1))
        mask[[0, 7, 16, 23, -1]] = 1
        if ax == 'horizontal':
            return median_filter(e_matrix, footprint=mask)
        elif ax == 'vertical':
            return median_filter(e_matrix, footprint=mask.T)

    def calc_anomaly_score(self, g):
        g = np.pad(g, ((0, 8 - g.shape[0] % 8), (0, 8 - g.shape[1] % 8)))
        blocks = split_into_blocks(g, (8, 8))

        max_1 = np.max(np.sum(blocks[:, 1:7, 1:7], axis=1), axis=1)
        max_2 = np.max(np.sum(blocks[:, 1:7, 1:7], axis=2), axis=1)
        min_1 = np.min(np.sum(blocks[:, 1:7, [0, 7]], axis=1), axis=1)
        min_2 = np.min(np.sum(blocks[:, [0, 7], 1:7], axis=2), axis=1)
        score = max_1 + max_2 - min_1 - min_2

        return score.reshape(g.shape[0]//8, g.shape[1]//8)



    def run(self):
        d_v, d_h = self.calc_approx_second_derivative()
        self.e_h = self.first_stage_BAG(d_h, 'horizontal')
        self.e_v = self.first_stage_BAG(d_v, 'vertical')
        self.g_h = self.second_stage_BAG(self.e_h, 'horizontal')
        self.g_v = self.second_stage_BAG(self.e_v, 'vertical')
        self.g = self.g_h + self.g_v
        self.score = self.calc_anomaly_score(self.g)
        return self.score




