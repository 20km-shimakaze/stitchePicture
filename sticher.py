import numpy as np
import cv2
import myutils


class Stitcher:
    # 将两个图片拼接
    def stitcher(self, image, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (img_b, img_a) = image
        (kps_a, features_a) = self.detect_describe(img_a)
        (kps_b, features_b) = self.detect_describe(img_b)
        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kps_a, features_a, kps_b, features_b, ratio, reprojThresh)
        if M is None:
            return None
        # H为3x3视角变换矩阵
        matches, H, status = M
        # 根据变换矩阵H将图片a进行视角变换，result是变换后的矩阵
        result = cv2.warpPerspective(img_a, H, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        # myutils.cv_show(result)
        result[0:img_b.shape[0], 0:img_b.shape[1]] = img_b
        # myutils.cv_show(result)
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(img_a, img_b, kps_a, kps_b, matches, status)
            # 返回结果
            return result, vis
            # 返回匹配结果
        return result

    def detect_describe(self, image):
        # 建立sift生成器
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features

    def matchKeypoints(self, kps_a, features_a, kps_b, features_b, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()
        # 使用knn匹配来自a,b图的sift特征匹配对，knn=2
        raw_matches = matcher.knnMatch(features_a, features_b, 2)
        # 过滤不要的特征匹配对
        matches = []
        for m in raw_matches:
            if len(m) and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # 最少要四对特征点才能计算矩阵
        if len(matches) > 4:
            pts_a = np.float32([kps_a[i] for (_, i) in matches])
            pts_b = np.float32([kps_b[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reprojThresh)
            return matches, H, status
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # 返回可视化结果
        return vis
