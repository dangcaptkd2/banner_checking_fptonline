
from flask_restful import Resource

from recognition import RECOGNITION_VN

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

recog = RECOGNITION_VN()

def processing(name=None):   
    result_vn = recog.get_result(name=name)
    print(">>>>> done recog VN")

    if result_vn is None:
        return {"text_vn": "None"}

    vietnam = []
    for k in result_vn:
        vietnam.append(result_vn[k])
        
    R = {"text_vn": ' '.join(vietnam)}
    return R

# if __name__ == '__main__': 
#     print(processing('21'))
#     print("Hello OCR")
#    