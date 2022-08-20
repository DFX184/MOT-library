import torch
import gdown
from pathlib import Path
import os
import numpy as np
import cv2 as cv
import torchvision
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

__models__ = {
    "bytetrack_x_mot17" : "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
    "bytetrack_l_mot17" : "1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
    "bytetrack_m_mot17" : "11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
    "bytetrack_s_mot17" : "1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
    "bytetrack_tiny_mot17" : "1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju",
    "bytetrack_x_mot20" : "1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U",
}
__files__ = {
    "bytetrack_x_mot17" :  "https://raw.githubusercontent.com/ifzhang/ByteTrack/main/exps/example/mot/yolox_x_mix_det.py", 
    "bytetrack_l_mot17" : "https://raw.githubusercontent.com/ifzhang/ByteTrack/main/exps/example/mot/yolox_l_mix_det.py",
    "bytetrack_m_mot17" : "https://raw.githubusercontent.com/ifzhang/ByteTrack/main/exps/example/mot/yolox_m_mix_det.py",
    "bytetrack_s_mot17" : "https://raw.githubusercontent.com/ifzhang/ByteTrack/main/exps/example/mot/yolox_s_mix_det.py"  ,
    "bytetrack_tiny_mot17" : "https://raw.githubusercontent.com/ifzhang/ByteTrack/main/exps/example/mot/yolox_tiny_mix_det.py" ,
    "bytetrack_x_mot20" : "https://raw.githubusercontent.com/ifzhang/ByteTrack/main/exps/example/mot/yolox_x_mix_det.py" 

}
__size__ = {
     "bytetrack_x_mot17" :(800,1440), 
    "bytetrack_l_mot17" : (800,1440),
    "bytetrack_m_mot17" :(800,1440) ,
    "bytetrack_s_mot17" :   (608, 1088),
    "bytetrack_tiny_mot17":(608, 1088),
    "bytetrack_x_mot20" :  (800,1440)
}

__RGB__ = {
    "bytetrack_x_mot17" : ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    "bytetrack_l_mot17" : ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    "bytetrack_m_mot17" : ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    "bytetrack_s_mot17" : ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    "bytetrack_tiny_mot17" : ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    "bytetrack_x_mot20" : ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

}


class YoloXInference(object):
    def __init__(self,model_name,ckpt_path = "./",device = "cpu",download = True, conf_thre=0.7, nms_thre=0.45):
        model_name = model_name.lower()
        if model_name not in __models__:
            raise RuntimeError(f"{model_name} not in {list(__models__.keys())}")
        self.model_name = model_name
        self.ckpt_path  = Path(ckpt_path)
        suffiex  = ".tar" if self.model_name == "bytetrack_x_mot20" else ".pth.tar"
        if not os.path.isfile(ckpt_path):
          if download or not (self.ckpt_path/(model_name + suffiex)).exists():
              if not (self.ckpt_path/(model_name + suffiex)).exists():
                  print(str(self.ckpt_path))
                  gdown.download(id=__models__[model_name],output = str(self.ckpt_path/(model_name + suffiex)),quiet=False)
          self.ckpt_path = self.ckpt_path/(model_name + suffiex)
        if not Path("./.exp").exists():
          os.mkdir("./.exp")
        version = self.model_name.split("_")[1]
        yolo    = f"yolo_{version}_mix_det.py"
        url     = __files__[self.model_name]
        output  = Path("./.exp")/yolo
        if not output.exists():
            torch.hub.download_url_to_file(url,output,progress=False)
        self.model = torch.hub.load("Megvii-BaseDetection/YOLOX","yolox_custom",
                        exp_path=output,
                        ckpt_path=self.ckpt_path)
        
        self.input_size = __size__[self.model_name]
        self.model = self.model.to(device)
        self.device= device
        self.mean,self.std = __RGB__[self.model_name]
        self.conf_thre = conf_thre
        self.nms_thre  = nms_thre
        self.model.eval()
    def preproc(self,image):
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(self.input_size) * 114.0

        img = np.array(image)
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if self.mean is not None:
            padded_img -= self.mean
        if self.std is not None:
            padded_img /= self.std
        padded_img = padded_img.transpose((2,0,1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self,prediction, num_classes=1):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self.nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output
        
    def __call__(self,image):
        img   = image.copy()
        

        img,_ = self.preproc(img)
        shape = img.shape
        img = torch.tensor(img) \
                  .unsqueeze(0) \
                  .to(self.device)

        with torch.no_grad():
            out = self.model(img)
            out = self.postprocess(out)[0].detach().cpu().numpy()
        boxes = out[:,:5]
        if (shape[0],shape[1]) != self.input_size:
          boxes[:,:4] = BoundingBoxesOnImage(
                  [
                      BoundingBox(x1=b[0],x2 = b[2],y1 = b[1],y2 = b[3])
                      for b in boxes
                  ],
                  shape = (self.input_size[0],self.input_size[1],3)
              ).on(image).to_xyxy_array()
        return boxes