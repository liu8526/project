from ai_hub import inferServer
import json
import base64
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.autograd import Variable as V
import base64

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

val_transform = A.Compose([
    # A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406, 0.38), std=(0.229, 0.224, 0.225, 0.38)),
    ToTensorV2(),
])


class seg_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model_name = 'efficientnet-b7'
        self.model = smp.UnetPlusPlus(          # UnetPlusPlus/DeepLabV3Plus
                encoder_name=self.model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=4,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=10,                # model output channels (number of classes in your dataset)
            )

    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x

class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)

    #数据前处理
    def pre_process(self, data):
        #json process
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        bytesIO = BytesIO()

        img = Image.open(BytesIO(bytearray(img)))
        img=np.array(img)
        img = img.astype(np.float32)

        transform=val_transform
        img = transform(image=img)['image']
        img=img.unsqueeze(0)
        return img
    
       #数据后处理
    def post_process(self, data):
        pred = data.squeeze().cpu().data.numpy()
        pred = np.argmax(pred,axis=0)
        pred = np.uint8(pred+1)

        # pred=Image.fromarray(pred)
        # pred=pred.convert('L')
        # # pred.save('./000006.png')
        # pred=np.asarray(pred)

        # print('pred:', pred)
        pred = cv2.imencode('.png', pred)[1]
        img_encode = np.array(pred).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data,'utf-8')
        return bast64_str

    #模型预测：默认执行self.model(preprocess_data)，一般不用重写,如需自定义，可覆盖重写
    def predict(self, data):
        with torch.no_grad():
            ret = self.model(data.cuda())
            return ret

if __name__ == "__main__":
    model=seg_model().cuda()
    model= torch.nn.DataParallel(model)
    checkpoints=torch.load('./SWA_checkpoint-best.pth')
    model.load_state_dict(checkpoints['state_dict'])
    model.eval()

    my_infer = myInfer(model)
    import logging
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename='my.log',
                    filemode='w')
    logging.info(my_infer.run(debuge=0))#默认为("127.0.0.1", 8080)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息