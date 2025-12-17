import cv2
import os
import numpy as np

class Getmask:
    def __init__(self, path):
        self.path = path
        self.dir = dir
        self.path_Real = path + 'BtoA(A)'
        self.path_Xray = path + 'BtoA(B)'

    def mask(self):
        print("Of course, I still love you")

        file_names_R = [f for f in os.listdir(self.path_Real) if f.endswith('.jpg') or f.endswith('.png')]
        for f in file_names_R:
            image_Real = cv2.imread(os.path.join(self.path_Real, f))
            image_Real = cv2.cvtColor(image_Real, cv2.COLOR_BGR2GRAY)
            t, rst = cv2.threshold(image_Real, 240, 255, cv2.THRESH_BINARY_INV)
            closing = cv2.morphologyEx(rst, cv2.MORPH_CLOSE, kernel = np.ones((5, 5), np.uint8))
            save_path = os.path.join(self.path, 'BtoA(Amask)')
            nohole = self.FillHole(closing, f, save_path)

        file_names_X = [f for f in os.listdir(self.path_Xray) if f.endswith('.jpg') or f.endswith('.png')]
        for f in file_names_X:
            image_XRay = cv2.imread(os.path.join(self.path_Xray, f))
            image_XRay = cv2.cvtColor(image_XRay, cv2.COLOR_BGR2GRAY)
            t, rst = cv2.threshold(image_XRay, 240, 255, cv2.THRESH_BINARY_INV)
            closing = cv2.morphologyEx(rst, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
            save_path = os.path.join(self.path, 'BtoA(Bmask)')
            nohole = self.FillHole(closing, f, save_path)

    def FillHole(self, im_in, f, SavePath):
        im_floodfill = im_in.copy()
        h, w = im_floodfill.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        isbreak = False
        for i in range(im_floodfill.shape[0]):
            for j in range(im_floodfill.shape[1]):
                if (im_floodfill[i][j] == 0):
                    seedPoint = (i, j)
                    isbreak = True
                    break
            if (isbreak):
                break
        print(f)
        cv2.floodFill(im_floodfill, mask, seedPoint, 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = im_in | im_floodfill_inv
        im_out = cv2.resize(im_out,(256, 256))
        cv2.imwrite(os.path.join(SavePath, f), im_out)
        print(im_out.shape)

        return im_out

if __name__ == '__main__':
    getmask = Getmask(path='../datasets/xray2bottle')
    getmask.mask()