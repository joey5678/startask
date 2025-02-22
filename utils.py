import cv2
import numpy as np

x0 = 0
y0 = 0
drawing = False


def mouse_handler(event, x, y, flags, data) :
    global x0, y0, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN :

        cv2.circle(data['im'], (x,y), 1, (0,0,255), 5, 16)

        x0, y0 = x, y
        cv2.imshow("Image", data['im'])
        if len(data['points']) < 4 :
            data['points'].append([x,y])

            
    if event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_RBUTTON:
        tmp_data = data['im'].copy()
        iH, iW = tmp_data.shape[:2]
        cv2.line(tmp_data, (x0, y0), (x, y), (255, 255, 0))
        cv2.line(tmp_data, (0, y), (iW, y), (0, 255, 0))
        cv2.line(tmp_data, (x, 0), (x, iH), (0, 255, 0))
        cv2.imshow("Image", tmp_data)

    elif event == cv2.EVENT_RBUTTONUP:
        cv2.imshow("Image", data['im'])


def get_four_points(im):
    global x0, y0
    x0, y0 = 0, 0

    data = {}
    data['im'] = im.copy()
    data['points'] = []

    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    if len(data['points']) != 4:
        return None
    points = np.vstack(data['points']).astype(float)

    return points


def eval_dst_size(pts_src):

    if abs(pts_src[1][0] - pts_src[0][0]) > 1.5 * abs(pts_src[1][1] - pts_src[0][1]):
        H = abs(max(pts_src[3][1], pts_src[2][1]) - max(pts_src[0][1], pts_src[1][1]))
        W = abs(max(pts_src[1][0], pts_src[2][0]) - max(pts_src[0][0], pts_src[3][0]))
    else:
        H = abs(max(pts_src[3][0], pts_src[2][0]) - max(pts_src[0][0], pts_src[1][0]))
        W = abs(max(pts_src[1][1], pts_src[2][1]) - max(pts_src[0][1], pts_src[3][1]))
    
    return H, W


def binaryzation(cv_img, reversed=False):

    gray_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2GRAY)
    if reversed:
        gray_img = 255 - gray_img
    
    blur_img = cv2.adaptiveThreshold(
                gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 20)

    img_denoised = cv2.fastNlMeansDenoising(blur_img, 10, 10, 7, 21)
    cv2.imwrite('img_denoised.png', img_denoised)

    return img_denoised


def save_marks_to_file(mlist, fname):
    file_name = fname + ".txt"
    with open(file_name, 'w') as f:
        for d in mlist:
            f.write('\t'.join([d['text'], d['ori'], str(d['pts'])]) + '\n')