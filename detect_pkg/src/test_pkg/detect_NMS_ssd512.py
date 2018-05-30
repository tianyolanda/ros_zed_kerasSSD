#!/usr/bin/env python3

from __future__ import print_function
import roslib
import sys
import math
import time
import cv2
import rospy
import std_msgs.msg
from new_detect.msg import OBJ
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import test_pkg.point_cloud2 as pcl2
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf

import random
import PIL as plt
import numpy as np

from keras import backend as K
from keras_ssd512 import ssd_512
import cv2

# slim = tf.contrib.slim

class image_converter: 

#zed_image = 0

    def __init__(self):

        self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=10)
        self.zed_image = None
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed/rgb/image_rect_color",Image,self.callback)

    def callback(self,data):
        #global zed_image
        cv_image= self.bridge.imgmsg_to_cv2(data)
        self.zed_image = cv_image

class image_converter_pointcloud:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_3",PointCloud2,queue_size=10)
        self.zed_image_pointcloud = None
        self.bridge = CvBridge()
        self.dict_1 = 0
        self.dict_2 = 0
        self.image_sub2 = rospy.Subscriber("/zed/point_cloud/cloud_registered",PointCloud2,self.callback_pointcloud)
        self.rate = rospy.Rate(1);
        # rospy.spin();

    def callback_pointcloud(self,data):
        # print('33333333333:icp.dict_1 = :',self.dict_1)    
        # print('33333333333:icp.dict_2 = :',self.dict_2) 
        data_out = pcl2.read_points(data,field_names = ("x", "y", "z"),skip_nans=False,uvs=[[self.dict_1,self.dict_2]])
        self.zed_image_pointcloud = data_out


def objRecord2Ros(class_name,score,cls_id,type_code,distance,x,y,z):
    # record the important(car&person) object to ROS msg
    global type_code_list,confidence_list,distance_list,x_list,y_list,z_list,objPerFrame
    confidence = round(score,2)
    objPerFrame = objPerFrame + 1

    #stuck into list for ROS transfer
    type_code_list[objPerFrame]=type_code
    confidence_list[objPerFrame]=confidence
    distance_list[objPerFrame]=round(distance,2)
    x_list[objPerFrame]=round(x,2)
    y_list[objPerFrame]=round(y,2)
    z_list[objPerFrame]=round(z,2)  

    print(class_name)
    print('confidence:',confidence)
    print('distance',distance_list[objPerFrame])
    print('x:',x_list[objPerFrame])
    print('y:',y_list[objPerFrame])
    print('z:',z_list[objPerFrame])

def calculateIoU(obj1, obj2):
    cx1 = obj1[0]
    cy1 = obj1[1]
    cx2 = obj1[2]
    cy2 = obj1[3]
 
    gx1 = obj2[0]
    gy1 = obj2[1]
    gx2 = obj2[2]
    gy2 = obj2[3]
 
    carea = (cx2 - cx1) * (cy2 - cy1) 
    garea = (gx2 - gx1) * (gy2 - gy1)
 
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h 
 
    iou = round(area / (carea + garea - area),2)
 
    return iou

def frameFilter(cls_id,xmin,xmax,ymin,ymax):
    global objPerlastFrame,total_objPerFrame,record_xmin,record_xmax,record_ymin,record_ymax,record_cls_id,record_obj_id
    global record_obj_num_appeared
    # record new frame object info
    record_xmin[1][total_objPerFrame] = xmin
    record_xmax[1][total_objPerFrame] = xmax
    record_ymin[1][total_objPerFrame] = ymin
    record_ymax[1][total_objPerFrame] = ymax
    record_cls_id[1][total_objPerFrame] = cls_id

    thisObjAppearedBefore = 0

    for i in range(objPerlastFrame):
        obj_record = [record_xmin[0][i+1],record_ymin[0][i+1],record_xmax[0][i+1],record_ymax[0][i+1]]
        obj_now = [xmin,ymin,xmax,ymax]
        if ((calculateIoU(obj_record,obj_now) > 0.6) and (cls_id == record_cls_id[0][i+1])):
            record_obj_id[1][total_objPerFrame] = record_obj_id[0][i+1]
            thisObjAppearedBefore = 1

    if thisObjAppearedBefore == 0 :
        record_obj_num_appeared = record_obj_num_appeared + 1
        record_obj_id[1][total_objPerFrame] = record_obj_num_appeared

    return thisObjAppearedBefore



def main():

    rospy.init_node('new_detect_pkg', anonymous=True)
    ic = image_converter()
    icp = image_converter_pointcloud()
    r = rospy.Rate(50)
    rospy.sleep(0.5)
    publisher()
    print('---------initialization model...please wait----------')
    # ssd_entity = SSD_entity()

    img_height = 512 # 272 Height of the input images
    img_width = 512 # 480 Width of the input images
    img_channels = 3 # Number of color channels of the input images

    # 1: Build the Keras model

    K.clear_session() # Clear previous models from memory.

    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                   two_boxes_for_ar1=True,
                   steps=[8, 16, 32, 64, 128, 256, 512],
                   offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   clip_boxes=False,
                   variances=[0.1, 0.1, 0.2, 0.2],
                   normalize_coords=True,
                   subtract_mean=[123, 117, 104],
                   swap_channels=[2, 1, 0],
                   confidence_thresh=0.5,
                   iou_threshold=0.45,
                   top_k=200,
                   nms_max_output_size=400)

    print("Model built.")

    # 2: Load the sub-sampled weights into the model.

    # Load the weights that we've just created via sub-sampling.
    # model.load_weights('/home/ogai1234/catkin_ws/src/detect_pkg/bin/ssd300_weights_epoch-45_loss-4.3010_val_loss-4.3788.h5', by_name=True)
    # model.load_weights('/home/ogai1234/catkin_ws/src/detect_pkg/bin/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5', by_name=True)
    model.load_weights('/home/ogai1234/catkin_ws/src/detect_pkg/bin/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5', by_name=True)

    print('---------model done----------')
    # model.load_weights(weights_path, by_name=True)

    # print("Weights file loaded:", weights_path)

    # classes = ["background", "dog", "umbrellaman", "cone", "car", "bicycle", "person"]
    classes = ['background','Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    key = ''
    t0 = time.time()
    frame_code = 1
    linewidth=3
    figsize=(10,10)
    # initilize the filter cach
    global record_xmin,record_xmax,record_ymin,record_ymax,record_cls_id,record_obj_id
    global record_obj_num_appeared, total_objPerFrame

    record_obj_num_appeared = 0

    record_xmin = [[0 for i in range(11)] for i in range(2)]
    record_xmax = [[0 for i in range(11)] for i in range(2)]
    record_ymin = [[0 for i in range(11)] for i in range(2)]
    record_ymax = [[0 for i in range(11)] for i in range(2)]
    record_cls_id = [[0 for i in range(11)] for i in range(2)]
    record_obj_id = [[0 for i in range(11)] for i in range(2)] 
    
    global objPerFrame
    objPerFrame=0
    total_objPerFrame = 0
    font = cv2.FONT_HERSHEY_TRIPLEX

    while (key != 113) and (not rospy.is_shutdown()):
        t1 = time.time()
        
        #one frame start
        frame_code = frame_code + 1      
        print("_________________________Frame:",frame_code)

        # for ros topic publish
        global objPerlastFrame,type_code_list,confidence_list,distance_list,x_list,y_list,z_list
        objPerlastFrame = total_objPerFrame
        # calculate the num of object appeared in one frame
        objPerFrame=0
        total_objPerFrame = 0

        # Ros message content initial
        type_code_list=[ 0 for i in range(11)]
        confidence_list=[ 0 for i in range(11)]
        distance_list=[ 0 for i in range(11)]
        x_list=[ 0 for i in range(11)]
        y_list=[ 0 for i in range(11)]
        z_list=[ 0 for i in range(11)]

        # load zed image from ros topic
        image = ic.zed_image # image.shape = (720, 1280, 3)     
        frame = image
        frame_old = frame
        # frame = frame[:,:,0:3]
        frame = cv2.resize(frame, (img_width,img_height))
        frame_np = np.array(frame)
        frame_np = frame_np[np.newaxis, :, :, :]
        
        # put frame into SSD network to do classification 
        y_pred = model.predict(frame_np)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
  
        colors = dict()

        for box in y_pred_thresh[0]:
            if objPerFrame > 9:
                break           
            cls_id = int(box[0])
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            # for person and car/train/bus, collect its information and transfer to ROS
            if cls_id == 15: #Person
                score = box[1]
                xmin = int(box[2] * frame_old.shape[1] / img_width)
                ymin = int(box[3] * frame_old.shape[0] / img_height)
                xmax = int(box[4] * frame_old.shape[1] / img_width)
                ymax = int(box[5] * frame_old.shape[0] / img_height)
                cv2.rectangle(frame_old, (xmin, ymin), (xmax, ymax),
                                         color=colors[cls_id],
                                         thickness=linewidth)
                # class_name = str(cls_id)
                # calculate distance and position 
                x_center = round((xmin+xmax) / 2)
                y_center = round((ymin+ymax) / 2)
                icp.dict_1=x_center
                icp.dict_2=y_center
                rospy.sleep(0.04)
                point_cloud = icp.zed_image_pointcloud
                x,y,z = 0,0,0
                #x,y,z is the position obtained from pointcloud2
                for p in point_cloud:
                    x,y,z = p
                    break
                distance = math.sqrt(x*x+y*y+z*z)
                type_code = cls_id
                class_name = classes[cls_id]
                # Person's typecode is 1
                type_code = 1
                total_objPerFrame = total_objPerFrame +1
                appeared = frameFilter(cls_id,xmin,xmax,ymin,ymax)
                if (appeared == 1):
                    objRecord2Ros(class_name,score,cls_id,type_code,distance,x,y,z)
                cv2.putText(frame_old, '{:s} | {:.2f} | {:.2f} |{:}'.format(class_name,score,distance,record_obj_id[1][total_objPerFrame]), (xmin, ymin+2), font, 0.5, (255, 255, 255), thickness=1)


            if cls_id == 7: #Car
                score = box[1]
                xmin = int(box[2] * frame_old.shape[1] / img_width)
                ymin = int(box[3] * frame_old.shape[0] / img_height)
                xmax = int(box[4] * frame_old.shape[1] / img_width)
                ymax = int(box[5] * frame_old.shape[0] / img_height)
                cv2.rectangle(frame_old, (xmin, ymin), (xmax, ymax),
                                         color=colors[cls_id],
                                         thickness=linewidth)
                # class_name = str(cls_id)
                # calculate distance and position 
                x_center = round((xmin+xmax) / 2)
                y_center = round((ymin+ymax) / 2)
                icp.dict_1=x_center
                icp.dict_2=y_center
                rospy.sleep(0.04)
                point_cloud = icp.zed_image_pointcloud
                x,y,z = 0,0,0
                #x,y,z is the position obtained from pointcloud2
                for p in point_cloud:
                    x,y,z = p
                    break
                distance = math.sqrt(x*x+y*y+z*z)
                type_code = cls_id
                class_name = classes[cls_id]
                # Car's typecode is 4
                type_code = 2
                total_objPerFrame = total_objPerFrame +1                
                appeared = frameFilter(cls_id,xmin,xmax,ymin,ymax)
                if (appeared == 1):
                    objRecord2Ros(class_name,score,cls_id,type_code,distance,x,y,z) 
                cv2.putText(frame_old, '{:s} | {:.2f} |{:}'.format(class_name,distance,record_obj_id[1][total_objPerFrame]), (xmin, ymin+2), font, 0.5, (255, 255, 255), thickness=1)
 
            if cls_id == 6: #Bus
                score = box[1]
                xmin = int(box[2] * frame_old.shape[1] / img_width)
                ymin = int(box[3] * frame_old.shape[0] / img_height)
                xmax = int(box[4] * frame_old.shape[1] / img_width)
                ymax = int(box[5] * frame_old.shape[0] / img_height)
                cv2.rectangle(frame_old, (xmin, ymin), (xmax, ymax),
                                         color=colors[cls_id],
                                         thickness=linewidth)
                # class_name = str(cls_id)
                # calculate distance and position 
                x_center = round((xmin+xmax) / 2)
                y_center = round((ymin+ymax) / 2)
                icp.dict_1=x_center
                icp.dict_2=y_center
                rospy.sleep(0.04)
                point_cloud = icp.zed_image_pointcloud
                x,y,z = 0,0,0
                #x,y,z is the position obtained from pointcloud2
                for p in point_cloud:
                    x,y,z = p
                    break
                distance = math.sqrt(x*x+y*y+z*z)
                type_code = cls_id
                class_name = classes[cls_id]
                # Car's typecode is 4
                type_code = 2
                total_objPerFrame = total_objPerFrame +1                
                appeared = frameFilter(cls_id,xmin,xmax,ymin,ymax)
                if (appeared == 1):
                    objRecord2Ros(class_name,score,cls_id,type_code,distance,x,y,z) 
                cv2.putText(frame_old, '{:s} | {:.2f} |{:}'.format(class_name,distance,record_obj_id[1][total_objPerFrame]), (xmin, ymin+2), font, 0.5, (255, 255, 255), thickness=1)
             
            if cls_id == 19: #Train
                score = box[1]
                xmin = int(box[2] * frame_old.shape[1] / img_width)
                ymin = int(box[3] * frame_old.shape[0] / img_height)
                xmax = int(box[4] * frame_old.shape[1] / img_width)
                ymax = int(box[5] * frame_old.shape[0] / img_height)
                cv2.rectangle(frame_old, (xmin, ymin), (xmax, ymax),
                                         color=colors[cls_id],
                                         thickness=linewidth)
                # class_name = str(cls_id)
                # calculate distance and position 
                x_center = round((xmin+xmax) / 2)
                y_center = round((ymin+ymax) / 2)
                icp.dict_1=x_center
                icp.dict_2=y_center
                rospy.sleep(0.04)
                point_cloud = icp.zed_image_pointcloud
                x,y,z = 0,0,0
                #x,y,z is the position obtained from pointcloud2
                for p in point_cloud:
                    x,y,z = p
                    break
                distance = math.sqrt(x*x+y*y+z*z)
                type_code = cls_id
                class_name = classes[cls_id]
                # Car's typecode is 4
                type_code = 4
                total_objPerFrame = total_objPerFrame +1                
                appeared = frameFilter(cls_id,xmin,xmax,ymin,ymax)
                if (appeared == 1):
                    objRecord2Ros(class_name,score,cls_id,type_code,distance,x,y,z) 
                cv2.putText(frame_old, '{:s} | {:.2f} |{:}'.format(class_name,distance,record_obj_id[1][total_objPerFrame]), (xmin, ymin+2), font, 0.5, (255, 255, 255), thickness=1)
 

            # if want show all classes, uncomment this line , and comment the upper 2 ifs
            # objRecord2Ros(class_name,score,cls_id,type_code,distance,x,y,z)
            # cv2.putText(frame, '{:s} | {:.2f} |{:}'.format(class_name,distance,record_obj_id[1][total_objPerFrame]), (xmin, ymin+2), font, 0.5, (255, 255, 255), thickness=1)

            # draw the bounding box
            # print format: class|conf|distance|x|y 
    
        # show the image with bounding box
        # image_back = cv2.resize(frame, (1080,720))
        cv2.imshow("image_back", frame_old)
        key  = cv2.waitKey(1)
        t21 = time.time()
        # calculate fps
        print('fps {:f}'.format( 1 / (t21 - t1)))
        talker()

        # last frame info move forward
        for i in range(10): 
            # print('record_xmin[0][i]:',record_xmin[0][i+1])
            # print('record_xmin[1][i]:',record_xmin[1][i+1])
            record_xmin[0][i+1] = record_xmin[1][i+1] 
            record_xmax[0][i+1] = record_xmax[1][i+1] 
            record_ymin[0][i+1] = record_ymin[1][i+1] 
            record_ymax[0][i+1] = record_ymax[1][i+1] 
            record_cls_id[0][i+1] = record_cls_id[1][i+1] 
            record_obj_id[0][i+1] = record_obj_id[1][i+1]
            record_xmin[1][i+1] = 0 
            record_xmax[1][i+1] = 0 
            record_ymin[1][i+1] = 0
            record_ymax[1][i+1] = 0
            record_cls_id[1][i+1] = 0 
            record_obj_id[1][i+1] = 0
        if record_obj_num_appeared > 10000:
            record_obj_num_appeared = 0
 


def talker():
    # type_ros=['type1']
    r = rospy.Rate(10) #10hz
    objinfo = OBJ()
    objinfo.objnum=objPerFrame
    objinfo.type1=type_code_list[1]
    objinfo.type2=type_code_list[2]
    objinfo.type3=type_code_list[3]
    objinfo.type4=type_code_list[4]
    objinfo.type5=type_code_list[5]
    objinfo.type6=type_code_list[6]
    objinfo.type7=type_code_list[7]
    objinfo.type8=type_code_list[8]
    objinfo.type9=type_code_list[9]
    objinfo.type10=type_code_list[10]

    objinfo.confidence1=confidence_list[1]
    objinfo.confidence2=confidence_list[2]
    objinfo.confidence3=confidence_list[3]
    objinfo.confidence4=confidence_list[4]
    objinfo.confidence5=confidence_list[5]
    objinfo.confidence6=confidence_list[6]
    objinfo.confidence7=confidence_list[7]
    objinfo.confidence8=confidence_list[8]
    objinfo.confidence9=confidence_list[9]
    objinfo.confidence10=confidence_list[10]

    objinfo.distance1=distance_list[1]
    objinfo.distance2=distance_list[2]
    objinfo.distance3=distance_list[3]
    objinfo.distance4=distance_list[4]
    objinfo.distance5=distance_list[5]
    objinfo.distance6=distance_list[6]
    objinfo.distance7=distance_list[7]
    objinfo.distance8=distance_list[8]
    objinfo.distance9=distance_list[9]
    objinfo.distance10=distance_list[10]

    objinfo.x1=x_list[1]
    objinfo.x2=x_list[2]
    objinfo.x3=x_list[3]
    objinfo.x4=x_list[4]
    objinfo.x5=x_list[5]
    objinfo.x6=x_list[6]
    objinfo.x7=x_list[7]
    objinfo.x8=x_list[8]
    objinfo.x9=x_list[9]
    objinfo.x10=x_list[10]
    
    objinfo.y1=y_list[1]
    objinfo.y2=y_list[2]
    objinfo.y3=y_list[3]
    objinfo.y4=y_list[4]
    objinfo.y5=y_list[5]
    objinfo.y6=y_list[6]
    objinfo.y7=y_list[7]
    objinfo.y8=y_list[8]
    objinfo.y9=y_list[9]
    objinfo.y10=y_list[10]

    objinfo.z1=z_list[1]
    objinfo.z2=z_list[2]
    objinfo.z3=z_list[3]
    objinfo.z4=z_list[4]
    objinfo.z5=z_list[5]
    objinfo.z6=z_list[6]
    objinfo.z7=z_list[7]
    objinfo.z8=z_list[8]
    objinfo.z9=z_list[9]
    objinfo.z10=z_list[10]

    obj_info_pub.publish(objinfo)


def publisher():
    global obj_info_pub
    obj_info_pub = rospy.Publisher('new_obj_info', OBJ,queue_size=10)


if __name__ == "__main__":
    main()
