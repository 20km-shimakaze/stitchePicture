import cv2


def cv_show(img, name='img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, weigh, height=None, inter=cv2.INTER_AREA):
    if height is None:
        h, w = image.shape[:2]
        # print(h, w)
        height = int(h * (weigh / w))
    return cv2.resize(image, (weigh, height), interpolation=inter)
