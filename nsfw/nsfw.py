from nsfw.nsfw_model import NSFW 
from nsfw.yolov5.detect import get_human, get_flag, get_weapon, get_crypto, get_boob
from nsfw.crop_human import human_filter, convert, convert_filter
# from nsfw.deepface import search_single_face

# from TextFuseNet.mid_process import 

import cv2 
from PIL import Image
import os

import gc 
import torch
import random 
import numpy as np

np.random.seed(42)

def draw_image(img_path, out_yolo):
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    for lst in out_yolo:
        cls, x1,x2,y1,y2 = convert(w, h, lst)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
    cv2.imwrite('./static/uploads/'+name, img)

def detect_flag(img_path, config):
    ban_list = ['ba_que', 'my', 'nga', 'trieu_tien', 'trung_quoc', 'ukraina', 'viet_nam',]
    # img_path = os.path.join(root_image_path, img_path)
    out_yolo = get_flag(img_path=img_path, config=config)
    if out_yolo is None:
        return False 
    if config["utils"]["draw"]:
        draw_image(img_path, out_yolo)

    names = [i[0] for i in out_yolo]
    ban_names = [i for i in names if i in ban_list]
    return ban_names

def detect_weapon(img_path, config):
    # img_path = os.path.join(root_image_path, img_path)
    out_yolo = get_weapon(img_path=img_path, config=config)
    if out_yolo is None:
        return [] 
    if config["utils"]["draw"]:
        draw_image(img_path, out_yolo)
    names = [i[0] for i in out_yolo]
    return names

def detect_crypto(img_path, config):
    # img_path = os.path.join(root_image_path, img_path)
    out_yolo = get_crypto(img_path=img_path, config=config)
    if out_yolo is None:
        return [] 
    if config["utils"]["draw"]:
        draw_image(img_path, out_yolo)
    names = [i[0] for i in out_yolo]
    return names

def detect_nsfw(img_path, config):
    path_save_sexy = config["path_save"]["sexy"]   # save images that model classify predict was sexy
    path_save_neural = config["path_save"]["neural"] # save images that model classify predict was neural
    path_save_sexy_but_not_has_boob = config["path_save"]["sexy_half"] #save images that model detect canot detect boob
    path_save_human4boob_detect = config["path_save"]["human4boob_detect"]  #save images to run model detect boob

    if not os.path.isdir(path_save_human4boob_detect):
        os.mkdir(path_save_human4boob_detect)
    if config['utils']['save_image']:
        if not os.path.isdir(path_save_neural):
            os.mkdir(path_save_neural)
        if not os.path.isdir(path_save_sexy):
            os.mkdir(path_save_sexy)
        if not os.path.isdir(path_save_sexy_but_not_has_boob):
            os.mkdir(path_save_sexy_but_not_has_boob)

    out_yolo = get_human(img_path=img_path, config=config)

    if out_yolo is None:
        return None
    img = cv2.imread(img_path)  
    h,w,_ = img.shape  
    
    image_draw = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    cordinates = 0
    cordinates = human_filter(w=w, h=h, lst=out_yolo, return_only_biggest_box=False)
    if not cordinates :
            return None
    for cordinate in cordinates:
        x1,x2,y1,y2 = cordinate
        if config['utils']['draw']:
            image_draw = cv2.rectangle(image_draw, (x1,y1), (x2,y2), (0,0,255), 1)
            name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
            cv2.imwrite('./static/uploads/'+name, image_draw)
        crop_rgb = img_rgb[y1:y2, x1:x2]
        crop_image = Image.fromarray(crop_rgb.astype('uint8'), 'RGB')
        
        if crop_image.size[0]*crop_image.size[1]>=config['threshold']['size_human']:
            model = NSFW(config=config)
            result_nsfw = model.predict(crop_image)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()

            if result_nsfw:
                result_boob = None
                if config['utils']['checking_boob']:
                    name = len(os.listdir(path_save_human4boob_detect))
                    path_ = "{}/{}.jpg".format(path_save_human4boob_detect, name)
                    crop_image.save(path_)
                    result_boob = get_boob(path_, config=config)
                    if config['utils']['verbose']:
                        print(">>> boob:", result_boob)
                    if result_boob is not None:
                        if config['utils']['save_image']:
                            tmp_name = len(os.listdir(path_save_sexy))
                            crop_image.save(f"{path_save_sexy}/{tmp_name}.jpg")

                        human_img_array = cv2.imread(path_)
                        h,w,_ = human_img_array.shape
                        boob_cor = convert_filter(w, h, result_boob)
                        for cor in boob_cor:
                            x1,x2,y1,y2 = cor
                            human_img_array = cv2.rectangle(human_img_array, (x1,y1), (x2,y2), (255,0,0), 1)
                            name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'__.jpg'
                        cv2.imwrite('./static/uploads/'+name, human_img_array)

                        return result_nsfw 
                    else:
                        if config['utils']['save_image']:
                            tmp_name = len(os.listdir(path_save_sexy_but_not_has_boob))
                            crop_image.save(f"{path_save_sexy_but_not_has_boob}/{tmp_name}.jpg")
                        if config['utils']['verbose']:
                            print(">>>>", "sexy but dont have boob")
                else:
                    if config['utils']['save_image']:
                        tmp_name = len(os.listdir(path_save_sexy))
                        crop_image.save(f"{path_save_sexy}/{tmp_name}.jpg")
                    return result_nsfw 
            else:
                if config['utils']['save_image']:
                    tmp_name = len(os.listdir(path_save_neural))
                    crop_image.save(f"{path_save_neural}/{tmp_name}.jpg")

    # cordinates = human_filter(w=w, h=h, lst=out_yolo, return_only_biggest_box=False)
    # print(">>> num human and hello", len(cordinates))
    # if cordinates:
    #     scores = []
    #     names = []
    #     xys = []
    #     threshold = 0.3
    #     for cor in cordinates:
    #         x1,x2,y1,y2 = cor
    #         crop_bgr = img[y1:y2, x1:x2]
    #         score, who = search_single_face(crop_bgr)
    #         scores.append(score)
    #         names.append(who)
    #         xys.append([x1,x2,y1,y2])
            
    #     if min(scores) < threshold:
    #         idx = np.argmin(scores)
    #         x1,x2,y1,y2 = xys[idx]
    #         color = (0, 255, 255)
    #         if draw:
    #             image_draw = cv2.rectangle(image_draw, (x1,y1), (x2,y2), color, 1)
    #             name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
    #             cv2.imwrite('./static/uploads/'+name, image_draw) 
    #         return names[idx]
    return False

if __name__ == "__main__":
    print(detect_nsfw('C:/Users/quyennt72/Desktop/nguyenxuanphuc2.jpg'))
        
    