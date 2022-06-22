"""
+Banner Checking pineline:

    Raw Image -> Sexy detection(CPU) -> Text detection(GPU) -> English text recognition(GPU)      ->   Result
                                                        -> Vietnamese text recognition(CPU)
    *note: due to conflict environment, Sexy detection and Vietnamese text recognition must get result throught API

+Detail:
    - Sexy detection module:
        >> Step 1: Yolov5 detect human, if 2 or more people in images, choose the biggest
        >> Step 2: resnet18 train classify with 2 class: sexy and neural
    - Text detection:
        >> Step 1: Use pre-train TextFuseNet to detect bbox
        >> Step 2: sort bbox follow its cordinate in other from left to right, top to bottom
        >> Save every bbox to serve Vietnamese text recognition API
    - English text recognition: Deeptext(ViSTR)
    - Vietnamese text recognition: VietOCR

+Result format:
{
    "Status": "Review" or "Block",              -> status of banner is review or block
    "ban keyword": [],                          -> list of key word detected base on list ban key word (brand-safe.xlsx)
    "sexy": "False" or "True",                  -> is banner have sexy concept or not 
    "text": "hello, anh yeu em",                -> text recognize through English text recognition module
    "text_vietnamese": "hello, anh yêu em",     -> text recognize through Vietnamese text recognition module
    "time_detect_sexy": 0.41704,                -> time run Sexy detection
    "time_detect_text": 0.96005,                -> time run Text detection
    "time_reg_eng": 0.26701,                    -> time run English text recognition 
    "time_reg_vn": 1.40674,                     -> time run Vietnamese text recognition API
    "time_reg_vn_in": 1.39848,                  -> time run Vietnamese text recognition in model
    "total_time": 3.07684                       -> total time to call banner checking api
"""


from flask_restful import Resource #, reqparse

from detection import DETECTION
from recognition import RECOGNITION 
from utils import mid_process, action_merge
from utils import check_text_eng, check_text_vi
from utils import call_api_vi, call_api_nsfw, check_is_vn, clear_folder

import torch

import os
import time
from collections import Counter

import torch
# from torch.autograd import Variable
import time

# from torch.utils.data import Dataset, DataLoader

import gc

path_image_root = './static/uploads/'
path_crop = './debugs/crop_images'
path_models_classify = './models/'


class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

def procssesing_image(filename=None):   
    clear_folder()
    start_time = time.time()

    R = {
        'text': None,
        'text_vietnamese': None,
        'time_detect_text': 0,
        'time_reg_eng': 0,
        'time_reg_vn': 0,
        'time_reg_vn_in': 0,
        'status_sexy': False, 
        'flag': False,
        'weapon': False,
        'crypto': False,
        'time_detect_image': 0,
        'Status': 'Review',
        'total_time': 0,
        'ban keyword': []
    }

    print(">>> file name:", filename)
    image_path = os.path.join(path_image_root, filename)
    print('>>> image path in process image:', image_path)
    name = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')
    print('>>> name file in process image:', name)

    ####################################### NOT SAFE FOR WORK MODULE
    start_nsfw = time.time()
    result_nsfw = call_api_nsfw(filename=image_path)

    print(">>>>> done nsfw")
    end_nsfw = time.time()

    R['status_sexy'] = result_nsfw['status_sexy']
    R['flag'] = result_nsfw['flag']
    R['status_face_reg'] = result_nsfw['status_face_reg']
    R['weapon'] = result_nsfw['weapon']
    R['crypto'] = result_nsfw['crypto']
    R['time_detect_image'] = round(end_nsfw-start_nsfw, 5)
    
    if R['status_sexy'] or R['flag'] or not R['status_face_reg'] is None or R['weapon'] or R['crypto']:
        R['Status'] = 'Block'
        R['total_time'] = round(time.time()-start_time,5)
        return R
    #====================================#

    ####################################### TEXT DETECTION M0DULE
    start_detect = time.time()
    detect = DETECTION()
    result_detect = detect.create_file_result(img_path=image_path, name=name)
    print(">>>>> done detect")

    end_detect = time.time()
    R['time_detect_text'] = round(end_detect-start_detect, 5)

    if not detect.ok:
        del detect
        gc.collect()
        torch.cuda.empty_cache()
        end_detect = time.time()
        R['total_time'] = round(time.time()-start_time,5)
        return R
    else:
        del detect
        gc.collect()
        torch.cuda.empty_cache()
        #====================================#

        ####################################### MID PROCESS - sort bbox, save bbox to image for vietnamese recognize
        start_mid = time.time()
        list_arr, sorted_cor = mid_process(name=name, path_image=image_path, result_detect=result_detect)
        print(">>>>> done mid process and helloooooooooooooo")
        print(">>>>> num boxes:", len(list_arr))
        end_mid = time.time()
        #====================================#

        ####################################### ENGLISH TEXT RECOGNIZE MODULE
        start_reg_eng = time.time()
        recog = RECOGNITION()
        result_eng = recog.predict_arr(bib_list=list_arr, name=name)
        print(">>>>> done recog ENG")

        english = []
        threshold = 0.6
        for k in result_eng[name]:
            if result_eng[name][k][1] >= threshold:
                english.append(result_eng[name][k][0])

        del recog
        gc.collect()
        torch.cuda.empty_cache()

        banned_eng = check_text_eng(' '.join(english))
        end_reg_eng = time.time()
        
        R['text'] = ' '.join(english)
        #====================================#
        ####################################### VIETNAMESE TEXT RECOGNIZE MODULE
        start_reg_vn = time.time()
        if not check_is_vn(english):    # check if possible is Vietnamese
            R["ban keyword"] = banned_eng
            R['time_reg_eng'] = round(end_reg_eng-start_reg_eng, 5)
            if len(banned_eng)>0:
                R['Status'] = 'Block'
                R['total_time'] = round(time.time()-start_time,5)
                return R
        else:
            action_merge(sorted_cor, name, image_path)
            print(">>> text is vietnamese")
            result_vi = call_api_vi(name=name)

            banned_vi = check_text_vi(result_vi["text_vn"])
            end_reg_vn = time.time()

            R['text_vietnamese'] = result_vi['text_vn']
            R['time_reg_vn'] = round(end_reg_vn-start_reg_vn, 5)
            R['time_reg_vn_in'] = result_vi['time_text_vn']
            R["ban keyword"] = banned_vi

            if len(banned_vi)>0:
                R['Status'] = 'Block'
                R['total_time'] = round(time.time()-start_time,5)
                return R
    R['total_time'] = round(time.time()-start_time,5)
    return R

####################################

# if __name__ == '__main__': 
#     #print(call_api_nsfw('D:/sources/banner_checking/ocr/TextFuseNet/static/uploads/9.png'))
#     # print("helloooooooooooooo")
#     print(procssesing_image('20.png'))
#     #print(call_api_vi('ID__13237'))
#     # print(call_api_vi('21'))
#     # print("Hello OCR")
# #    