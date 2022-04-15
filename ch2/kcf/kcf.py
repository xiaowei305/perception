import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real

class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding = (0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)

class Tracker():
    def __init__(self):
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.debug = True

    def get_feature(self, image, roi):
        cx, cy, w, h = roi
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        sub_image = image[y:y+h, x:x+w, :]
        resized_image = cv2.resize(sub_image, (self.pw, self.ph))

        feature = self.hog.get_feature(resized_image)
 
        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2*np.pi*hann1t / (fw-1)))
        hann2t = 0.5 * (1 - np.cos(2*np.pi*hann2t / (fh-1)))
        hann2d = hann2t * hann1t

        feature = feature * hann2d

        if self.debug:
            self.sub_image = sub_image
            self.feature = feature
        return feature

    def gaussian_peak(self, w, h):
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh+h, -sxh:-sxh+w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x**2 + y**2)/(2. * sigma**2)))
        return g

    def train(self, x, y, sigma, lambdar):
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, alphaf, x, z, sigma):
        k = self.kernel_correlation(x, z, sigma)
        return real(ifft2(self.alphaf * fft2(k)))

    def kernel_correlation(self, x1, x2, sigma):
        c = ifft2(np.sum(conj(fft2(x1)) * fft2(x2), axis=0))
        c = fftshift(c)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def init(self, image, roi):
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[2], x.shape[1])
        if self.debug:
            self.y = y
        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def visualize(self, res):
        fh, fw = res.shape
        image = np.ones((fh + 20, fw * 4, 3), dtype=np.uint8) * 255
        image[:fh, :fw] = cv2.resize(self.sub_image, (fw, fh))
        image_hog = np.uint8(self.feature[4:7, :, :].transpose(1, 2, 0) * 10000)
        image[:fh, fw:fw*2] = image_hog
        image[:fh, fw*2:fw*3] = np.uint8(self.y * 10000)[:, :, None]
        image[:fh, fw*3:fw*4] = np.uint8((res- res.min()) * 10000)[:, :, None]
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, "IMAGE", (5, fh+13), font, 1, 255, 1)
        cv2.putText(image, "FEATURE", (fw+5, fh+13), font, 1, 255, 1)
        cv2.putText(image, "Y", (fw*2+35, fh+13), font, 1, 255, 1)
        cv2.putText(image, "RESULT", (fw*3+5, fh+13), font, 1, 255, 1)
        cv2.imshow("kcf", image)
        if not hasattr(self, "paused"):
            cv2.waitKey(0)
            self.paused = True
        else:
            cv2.waitKey(1)

    def update(self, image):
        cx, cy, w, h = self.roi
        max_response = -1
        for scale in [0.95, 1.0, 1.05]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            z = self.get_feature(image, roi)
            responses = self.detect(self.alphaf, self.x, z, self.sigma)
            height, width = responses.shape
            if self.debug and scale == 1.0:
                self.visualize(responses)
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        self.roi = (cx + dx, cy + dy, best_w, best_h)
        #update template
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return (cx - w // 2, cy - h // 2, w, h)
