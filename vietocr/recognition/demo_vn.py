from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import os
from PIL import Image

# sys.path.append()

class RECOGNITION_VN():
    def __init__(self) -> None:
        self.path_folder_crop_images = '../debugs/crop_images/'
        
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = './models/transformerocr.pth'
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cpu'
        self.config['predictor']['beamsearch']=False

        self.model = None

    def get_model(self):
        if self.model is not None:
            return self.model 
        
        self.model = Predictor(self.config)

        return self.model
    def predict(self, image_path):
        model = self.get_model()
        img = Image.open(image_path)
        return model.predict(img, return_prob=True)
    
    def get_result(self, name):
        crop_folder = self.path_folder_crop_images + name
        if len(os.listdir(crop_folder)) == 0 or not os.path.isdir(crop_folder):
            return None

        result = {}
        for img_name in os.listdir(crop_folder):
            img_path = os.path.join(crop_folder, img_name)
            tmp, score = self.predict(img_path)
            print(">>> vn:", tmp, score)
            if score>0.7:
                id = int(img_name.split('.')[0])
                result[id] = tmp

        return result

# if __name__ == '__main__': 
#     a = RECOGNITION_VN()

#     r = a.get_result(name = '21')

#     print(r)
