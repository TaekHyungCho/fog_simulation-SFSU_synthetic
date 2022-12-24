import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
#from source import gf
import time
from tqdm import tqdm
import skimage.measure

def transmission_homogeneoous_medium(depth_map,camera_parameter_files,beta):
    
    with open(camera_parameter_files) as json_file:
        cam_params = json.load(json_file)
        f_x = cam_params['intrinsic']['fx']
        f_y = cam_params['intrinsic']['fy']
        u_0 = cam_params['intrinsic']['u0']
        v_0 = cam_params['intrinsic']['v0']
    json_file.close()

    height, width = depth_map.shape[0], depth_map.shape[1]
    x_ = np.linspace(1,1,width)
    y_ = np.linspace(1,1,height)
    X,Y = np.meshgrid(x_,y_)
    matrix_ = np.sqrt((np.power(f_x,2) + np.power((X-u_0),2) + np.power((Y-v_0),2))/np.power(f_x,2))
    distance_map_in_meters = depth_map * matrix_

    t_initial = np.exp(-beta * distance_map_in_meters)
    return t_initial

def get_dark_channel(clear_image,window_size):
    kernel = np.ones((window_size,window_size),np.uint8)
    I_eroded = cv2.erode(clear_image,kernel)
    I_dark = np.min(I_eroded,axis=-1)
    return I_dark

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols

def  estimage_atmospheric_light_rf(I_dark,I,grayImg):
    brightest_pixel_fraction = 0.001
    height, width = I_dark.shape[0], I_dark.shape[1]
    num_of_pixel = height*width
    brightest_pixels_count_tmp = np.floor(brightest_pixel_fraction*num_of_pixel)
    brightest_pixels_count = int(brightest_pixels_count_tmp + np.mod(brightest_pixels_count_tmp+1,2))
    I_dark_vector = np.reshape(I_dark,(num_of_pixel))
    indices = np.flip(np.argsort(I_dark_vector))
    brightest_pixels_indices = indices[:brightest_pixels_count]
    
    I_gray_vector =np.reshape(grayImg,(num_of_pixel))
    I_gray_vector_brightest_pixels = I_gray_vector[brightest_pixels_indices]
    
    median_intensity = np.median(I_gray_vector_brightest_pixels)
    index_median_intensity = np.argwhere(I_gray_vector_brightest_pixels==median_intensity)[0]
    index_L = brightest_pixels_indices[index_median_intensity[0]]
    rows_L, cols_L = ind2sub((height,width),index_L)

    L = I[rows_L,cols_L,:]

    return L

def estimate_local_light_rf(I_dark,kernel_size):
    height, width = I_dark.shape[0], I_dark.shape[1]
    I_ = skimage.measure.block_reduce(I_dark,(kernel_size,kernel_size),np.max)
    I_resized = cv2.resize(I_,(width,height))
    I_filtered = cv2.GaussianBlur(I_resized,(0,0),9)
    return I_filtered

def haze_linear(clear_img,t,L):
    ch_ = L.shape[0]
    t = np.expand_dims(t,axis=-1)
    t_replicated = np.tile(t,(1,1,ch_))
    #t_shape = t.shape
    img = t_replicated * clear_image + (1-t_replicated) * np.tile(L,t.shape)
    return img

def haze_linear2(clear_image,t,L):
    t = np.expand_dims(t,axis=-1)
    L = np.expand_dims(L,axis=-1)
    t_replicated = np.tile(t,(1,1,3))
    L_ = np.concatenate((L,L,L),axis=-1)
    img = t_replicated * clear_image + (1-t_replicated) * L_
    gamma = 1/1.2
    img = np.power(img,gamma)
    return img 


if __name__ == '__main__':
    img_clear_list =[]
    image_path = "/data1/cth/nuscenes/"
    with open('/data1/cth/nuscenes/val_img.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            #data_path = image_path + line
            img_clear_list.append(line)
    depth_path_list = ['depth/CAM_FRONT/','depth/CAM_FRONT_RIGHT/','depth/CAM_FRONT_LEFT/',
                    'depth/CAM_BACK/','depth/CAM_BACK_LEFT/','depth/CAM_BACK_RIGHT/']
    kernel_size = 25
    for img in tqdm(img_clear_list):
        file_name = str(img).split('/')[-1]

        if '__CAM_FRONT__' in file_name:
            depth_path = image_path + depth_path_list[0]
            camera_parameter_files = "data/demos/camera/CAM_FRONT.json"
            save_detail = "CAM_FRONT/"
        elif '__CAM_FRONT_RIGHT__' in file_name:
            depth_path = image_path + depth_path_list[1]
            camera_parameter_files = "data/demos/camera/CAM_FRONT_RIGHT.json"
            save_detail = "CAM_FRONT_RIGHT/"
        elif '__CAM_FRONT_LEFT__' in file_name:
            depth_path = image_path + depth_path_list[2]
            camera_parameter_files = "data/demos/camera/CAM_FRONT_LEFT.json"
            save_detail = "CAM_FRONT_LEFT/"
        elif '__CAM_BACK__' in file_name:
            depth_path = image_path + depth_path_list[3]
            camera_parameter_files = "data/demos/camera/CAM_BACK.json"
            save_detail = "CAM_BACK/"
        elif '__CAM_BACK_LEFT__' in file_name:
            depth_path = image_path + depth_path_list[4]
            camera_parameter_files = "data/demos/camera/CAM_BACK_LEFT.json"
            save_detail = "CAM_BACK_LEFT/"
        elif '__CAM_BACK_RIGHT__' in file_name:
            depth_path = image_path + depth_path_list[5]
            camera_parameter_files = "data/demos/camera/CAM_BACK_RIGHT.json"
            save_detail = "CAM_BACK_RIGHT/"
        
        depth_map_name = depth_path + file_name.replace("jpg","tif")
        #clear_image = np.array(cv2.imread(img),dtype=np.float64) #BGR format
        clear_image = cv2.imread(img)
        grayImg = cv2.cvtColor(clear_image,cv2.COLOR_BGR2GRAY)
        clear_image = np.array(clear_image,dtype=np.float64)/255
        grayImg = np.array(grayImg,dtype=np.float64)/255
        depth_map = np.array(Image.open(depth_map_name))
        depth_map = (1-depth_map) 
        max_d = np.amax(depth_map)
        min_d = np.amin(depth_map)
        depth_map = (depth_map - min_d) / (max_d - min_d) * 51.2

        #result_root_path = "/home/cth/Hyundai_/src/futr3d_thcchhoo/"
        result_foggy_path = "/data1/cth/nuscenes/foggy/"
        save_path = result_foggy_path + save_detail
        #save_path = result_foggy_path 
        #Fog thickness
        beta = 0.02

        t_initial = transmission_homogeneoous_medium(depth_map,camera_parameter_files,beta)
        #t_ = gf.guided_filter(clear_image,t_initial,r=41,eps=1e-3)

        clear_image_dark_channel = get_dark_channel(clear_image,window_size=15)
        #L_atm = estimage_atmospheric_light_rf(clear_image_dark_channel,clear_image,grayImg)
        L_atm = estimage_atmospheric_light_rf(clear_image_dark_channel,clear_image,grayImg)
        foggy_image = haze_linear(clear_image,t_initial,L_atm)*255
        save_name = save_path + file_name
        cv2.imwrite(save_name,foggy_image)
        
        




        




        