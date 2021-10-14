#!/usr/bin/env python
#encoding: utf8

import numpy as np
import math
import lidar_to_grid_map as lg
import rospy
import tf
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rl_costmap_msgs.msg import TimeSeriesImages
from rl_costmap_msgs.srv import StateTSImagesGeneration
from rl_msgs.srv import StateImageGenerationSrv
#from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
#from std_srvs.srv import SetBool
#from rl_msgs.srv import MergeScans
#import time

class LocalCostmapGenerator():
    def __init__(self):
        #rospy.Subscriber("/chatter", String, self.get_img_callback) # テスト用
        
        img_service_name = rospy.get_name() + "/get_image"
        #rospy.Service(img_service_name, StateTSImagesGeneration, self.get_img_callback)
        rospy.Service(img_service_name, StateImageGenerationSrv, self.get_img_callback)

        ns = rospy.get_namespace()
        self.img_width = rospy.get_param("%srl_agent/img_width_pos"%ns) + rospy.get_param("%s/rl_agent/img_width_neg"%ns)
        self.img_height = rospy.get_param("%srl_agent/img_height"%ns)
        self.range_max, self.range_min = 4.0, -4.0
        self.xy_reso = (self.range_max - self.range_min) / self.img_width

        self.og_pub = rospy.Publisher('%srl_map'%ns, OccupancyGrid, queue_size=1)

    def get_img_callback(self, req):
        rospy.loginfo(rospy.get_caller_id()+"I receive scan & waypoints") # テスト用
        imgs = TimeSeriesImages()
        imgs.header.stamp = rospy.Time.now()
        
        d = 2          # t=0,…,dまでのマップを作成
        ped_num = 2    # 歩行者の人数 reqで受け取れるといい
        delta_t = 0.1  # タイムステップの間隔 書き換える


        # 歩行者経路予測  これもreqで受け取りたい
        #pred_ped_traj = np.empty((ped_num, 2, 2))  # 人数, 予測ステップ数, ローカル座標
        #for i in range(ped_num):
        #    pred_ped_traj[i] = predict_ped_traj(d, delta_t, ped_traj[i])


        # 時系列マップ生成
        t0_map = self.generate_costmap(req.scan, req.wps)
        self.og_pub.publish(t0_map) # debug
        imgs.maps.append(t0_map)
        
        # 歩行者まだ
        #imgs.img_t1 = generate_costmap(img_size, req.scan, req.wp, pred_ped_traj[:,0])
        #imgs.img_t2 = generate_costmap(img_size, req.scan, req.wp, pred_ped_traj[:,1])
        
        #return imgs
        return t0_map

    def add_scan_to_image(self, image, scan):
        occupied = 100
        free = 30
        
        center_ix, center_iy = int(self.img_width/2), int(self.img_height/2)  # center coordinate of the grid map

        for i in range(len(scan.ranges)):
            ang = scan.angle_min + scan.angle_increment
            dist = scan.ranges[i]
            if np.isnan(dist) or dist == 0.0:
                continue
            else:
                x = np.sin(ang) * dist
                y = np.cos(ang) * dist

            # occupancy grid computed with bresenham ray casting
            ix = int(round((x - self.range_min) / self.xy_reso))
            iy = int(round((y - self.range_min) / self.xy_reso))
            laser_beams = lg.bresenham((center_ix, center_iy), (ix, iy))  # line form the lidar to the occupied point

            for laser_beam in laser_beams:
                idx = self.index_2d_to_1d(laser_beam[0],laser_beam[1])
                if 0 <= idx < self.img_width*self.img_height:
                    image[idx] = free # free area

            if 0 <= self.index_2d_to_1d(ix, iy) < self.img_width*self.img_height:
                image[self.index_2d_to_1d(ix, iy)] = occupied  # occupied area
            if 0 <= self.index_2d_to_1d(ix+1, iy) < self.img_width*self.img_height:
                image[self.index_2d_to_1d(ix+1, iy)] = occupied  # extend the occupied area
            if 0 <= self.index_2d_to_1d(ix, iy+1) < self.img_width*self.img_height:
                image[self.index_2d_to_1d(ix, iy+1)] = occupied  # extend the occupied area
            if 0 <= self.index_2d_to_1d(ix+1, iy+1) < self.img_width*self.img_height:
                image[self.index_2d_to_1d(ix+1, iy+1)] = occupied  # extend the occupied area

    def add_path_to_image(self, image, wp):
        path = 0  # grid value for path
        center_ix, center_iy = int(self.img_width/2), int(self.img_height/2)  # center coordinate of the grid map
        prev_ix, prev_iy = center_ix, center_iy
        for p in wp.points:
            ix = int(round((p.x - self.range_min) / self.xy_reso))
            iy = int(round((p.y - self.range_min) / self.xy_reso))
            laser_beams = lg.bresenham((prev_ix, prev_iy), (ix, iy))
            for laser_beam in laser_beams:
                idx = self.index_2d_to_1d(laser_beam[0],laser_beam[1])
                if 0 <= idx < self.img_width*self.img_height:
                    image[idx] = path
            prev_ix, prev_iy = ix, iy

    def index_2d_to_1d(self, x_index, y_index):
        index = y_index * self.img_width + x_index
        if x_index < 0 or x_index >= self.img_width or y_index < 0 or y_index > self.img_height:
            index = self.img_width * self.img_height + 1
    
        return index

    # 各時刻のローカルマップ生成
    def generate_costmap(self, scan, wp, pred_ped_pos=None):
        # set info
        img = OccupancyGrid()
        img.header.stamp = rospy.Time.now()
        img.header.frame_id = "/map"
        img.info.resolution = self.xy_reso
        img.info.width = self.img_width
        img.info.height = self.img_height
        img.info.origin.position.x = 0.0
        img.info.origin.position.y = -self.img_height * self.xy_reso / 2.0

        img_data = np.ones(self.img_width * self.img_height) * 50

        # スキャンデータを占有グリッドマップに変換
        self.add_scan_to_image(img_data, scan)

        # ゴールまでの経路をoccupancy_mapに追加
        self.add_path_to_image(img_data, wp)

        # 歩行者予測位置をローカル座標系に変換
        # 予測位置の埋め込み
        ''' if not pred_ped_pos is None:
            for (x, y) in pred_ped_pos:
                print(x,y)
                ix = int(round((x - min_x) / xy_reso))
                iy = int(round((y - min_y) / xy_reso))
                img_data[ix][iy] = 0.0  # 歩行者予測位置は黒
'''
   
        img.data = img_data 
        return img


# あとで別のノードにする
# dステップ先までの歩行者の予測位置を返す
# 歩行者が観測時と同速度で移動する前提
def predict_ped_traj(d, delta_t, traj):
    # traj: 1つ前と1つ先の歩行者の位置

    pred_traj = np.empty((d,2))
    
    for j in range(d):
        v = (traj[1] - traj[0])/delta_t
        pred_traj[j] = traj[1] + (j+1) * v * delta_t

    return pred_traj


if __name__ == '__main__':
    rospy.init_node('costmap_generator')
    lcg = LocalCostmapGenerator()
    rospy.spin()
