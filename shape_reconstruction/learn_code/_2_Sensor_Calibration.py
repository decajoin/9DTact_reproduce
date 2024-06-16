# collect images and calibrate the sensor
import numpy as np
import cv2
import yaml
import os
from shape_reconstruction import Sensor


class SensorCalibration:
    def __init__(self, cfg_path):
        f = open(cfg_path, 'r+', encoding='utf-8')
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.sensor = Sensor(cfg, calibrated=False)
        if not os.path.exists(self.sensor.depth_calibration_dir):
            os.makedirs(self.sensor.depth_calibration_dir)
        depth_calibration = cfg['depth_calibration']
        self.BallRad = depth_calibration['BallRad']
        self.circle_detection_gray = depth_calibration['circle_detect_gray']
        self.show_circle_detection = depth_calibration['show_circle_detection']

    def run(self):
        print("DON'T touch the sensor surface!!!!!")
        print('Please press "y" to save the reference image!')
        ref = self.sensor.get_raw_avg_image()
        cv2.imwrite(self.sensor.depth_calibration_dir + '/ref.png', ref)
        rc_ref = self.sensor.rectify_crop_image(ref)
        cv2.imwrite(self.sensor.depth_calibration_dir + '/rectify_crop_ref.png', rc_ref)
        print('Reference image saved!')
        print('Please press the ball on the sensor and press "y" to save the sample image!')
        sample = self.sensor.get_raw_avg_image()
        cv2.imwrite(self.sensor.depth_calibration_dir + '/sample.png', sample)
        rc_sample = self.sensor.rectify_crop_image(sample)
        cv2.imwrite(self.sensor.depth_calibration_dir + '/rectify_crop_sample.png', rc_sample)
        print('Sample image saved!')

        gray_list, depth_list = self.mapping_data_collection(rc_sample, rc_ref)
        gray_list = np.array(gray_list)
        depth_list = np.array(depth_list)
        Pixel_to_Depth = self.get_list(gray_list, depth_list)
        np.save(self.sensor.Pixel_to_Depth_path, Pixel_to_Depth)

    def circle_detection(self, diff):
        ## 灰度图
        diff_gray = (diff[::, ::, 0] + diff[::, ::, 1] + diff[::, ::, 2]) / 3
        ## 二值图
        contact_mask = (diff_gray > self.circle_detection_gray).astype(np.uint8)
        ## 轮廓检测算法，输出轮廓列表
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ## 计算每个轮廓的面积，并输出面积列表
        areas = [cv2.contourArea(c) for c in contours]
        ## 对面积从小到大排序
        sorted_areas = np.sort(areas)
        if len(sorted_areas):
            ## 找出面积最大的轮廓
            cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
            ## 返回最小外切圆的圆心和半径
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if self.show_circle_detection:
                """
                显示所检测到的外切圆，并且可以手动调整圆心位置和半径大小，默认为False
                """
                key = -1
                print('If the detected circle is suitable, press the key "q" to continue!')
                while key != ord('q'):
                    center = (int(x), int(y))
                    radius = int(radius)
                    circle_show = cv2.circle(np.array(diff), center, radius, (0, 255, 0), 1)
                    circle_show[int(y), int(x)] = [255, 255, 255]
                    cv2.imshow('contact', circle_show.astype(np.uint8))
                    key = cv2.waitKey(0)
                    if key == ord('w'):
                        y -= 1
                    elif key == ord('s'):
                        y += 1
                    elif key == ord('a'):
                        x -= 1
                    elif key == ord('d'):
                        x += 1
                    elif key == ord('m'):
                        radius += 1
                    elif key == ord('n'):
                        radius -= 1
                cv2.destroyWindow('contact')
            return center, radius
        else:
            return (0, 0), 0

    def mapping_data_collection(self, img, ref):
        gray_list = []
        depth_list = []
        diff_raw = ref - img
        ## 创建掩膜
        ## 将diff中小于150的值置为0，大于150的值置为1；即像素明度小于150的点置为1，大于150的点置为0
        diff_mask = (diff_raw < 150).astype(np.uint8)
        diff = diff_raw * diff_mask
        cv2.imshow('ref', ref)
        cv2.imshow('img', img)
        cv2.imshow('diff', diff)
        ## 存入检测接触轮廓外切圆的圆心位置和半径
        center, detect_radius_p = self.circle_detection(diff)
        if detect_radius_p:
            ## 生成像素等距数组：start，end，num
            x = np.linspace(0, diff.shape[0] - 1, diff.shape[0])  # [0, 479]
            y = np.linspace(0, diff.shape[1] - 1, diff.shape[1])  # [0, 639]
            ## 用y和x生成网格。
            xv, yv = np.meshgrid(y, x)
            ## 计算出圆心到网格中每个点(每个像素)的距离，存入rv
            xv = xv - center[0]
            yv = yv - center[1]
            rv = np.sqrt(xv ** 2 + yv ** 2)
            ## mask用来取出投影圆内的像素点
            mask = (rv < detect_radius_p)
            ## 计算圆内像素点距离圆心的距离(mm)的平方，存入temp
            temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * self.sensor.pixel_per_mm ** 2
            ## 勾股定理，计算出圆内像素点距离圆心的高度(mm)，再减去按压投影园的半径，即得到该像素的按压深度，存入height_map
            height_map = (np.sqrt(self.BallRad ** 2 - temp) * mask - np.sqrt(
                self.BallRad ** 2 - (detect_radius_p * self.sensor.pixel_per_mm) ** 2)) * mask
            ## 将nan改为0避免运算错误
            height_map[np.isnan(height_map)] = 0
            ## 获取灰度图
            diff_gray = (diff[::, ::, 0] + diff[::, ::, 1] + diff[::, ::, 2]) / 3
            # diff_gray = self.sensor.crop_image(diff_gray)
            # height_map = self.sensor.crop_image(height_map)
            count = 0
            for i in range(height_map.shape[0]):
                for j in range(height_map.shape[1]):
                    """
                    将深度值大于0的像素点的按压深度和灰度值对应起来
                    """
                    if height_map[i, j] > 0:
                        gray_list.append(diff_gray[i, j])
                        depth_list.append(height_map[i, j])
                        count += 1
            print('Sample points number: {}'.format(count))
            return gray_list, depth_list

    def get_list(self, gray_list, depth_list):
        """
        生成一个灰度值与深度值对应关系的数组
        """
        ## 以最大的灰度值作为范围
        GRAY_scope = int(gray_list.max())
        ## 生成一个0～GRAY_scope的数组
        GRAY_Height_list = np.zeros(GRAY_scope + 1)
        for gray_number in range(GRAY_scope + 1):
            """
            遍历0～GRAY_scope的灰度值，计算出每个灰度值对应的平均深度值
            """
            gray_height_sum = depth_list[gray_list == gray_number].sum()
            gray_height_num = (gray_list == gray_number).sum()
            if gray_height_num:
                GRAY_Height_list[gray_number] = gray_height_sum / gray_height_num
        for gray_number in range(GRAY_scope + 1):
            if GRAY_Height_list[gray_number] == 0:
                if not gray_number:
                    """
                    如果灰度值为0，则min为-1,max为1；遍历此灰度值后面的灰度值，直到找到高度不为0的灰度值，将此索引作为max
                    计算平均深度值
                    """
                    min_index = gray_number - 1
                    max_index = gray_number + 1
                    for i in range(GRAY_scope - gray_number):
                        if GRAY_Height_list[gray_number + 1 + i] != 0:
                            max_index = gray_number + 1 + i
                            break
                    GRAY_Height_list[gray_number] = (GRAY_Height_list[max_index] - GRAY_Height_list[min_index]) / (
                            max_index - min_index)
        return GRAY_Height_list


if __name__ == '__main__':
    config_path = 'shape_config.yaml' 
    depth_calibration = SensorCalibration(config_path)
    depth_calibration.run()
