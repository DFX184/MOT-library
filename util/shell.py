from pathlib import Path
from pytube import YouTube
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2 as cv
from moviepy.editor import *
from tqdm.auto import tqdm
from alfred.vis.image.det import visualize_det_cv2_part
import numpy as np
from . import metric

def read_rgb(path):
    img = cv.imread(path)
    return cv.cvtColor(img,cv.COLOR_BGR2RGB)


def run(
    tracker,
    video_url_filenames,
    output_video,
    class_names,
    resize  = None,
    logname = None,
    scale   = 1,
    fps     = 25,
    gif     = False
    ):
    
    if not isinstance(video_url_filenames,(str,list,np.ndarray)):
        raise RuntimeError(
            "`video_path_fileanames` must be one of ['video_path','image filenames','url']"
        )
    _extension = [
        ".mp4",
    ]
    frames = []
    if isinstance(video_url_filenames,(list,np.ndarray)):
        bar = tqdm(video_url_filenames)
    elif not Path(video_url_filenames).suffix  in extension:
        bar = tqdm(VideoFileClip(video_url_filenames).iter_frames())
    else:
        filepath = YouTube(video_url_filenames).streams.filter(progressive=True, file_extension='mp4') \
                                                                     .order_by('resolution').desc().first().download()
        bar = tqdm(VideoFileClip(filepath).iter_frames())
    
    
    if not logname is None:
        f = open(logname,"w")
    frame_id = 1
    for file in bar:
        if not isinstance(video_url_filenames,str):
            img    = read_rgb(file)
        else:
            img    = file
        if not resize is None:
            img_r       = cv.resize(img,tuple(resize))
            boxes       = tracker.update(img_r)
            boxes[:,1:5]= BoundingBoxesOnImage(
                [
                    BoundingBox(x1=b[1],x2 = b[3],y1 = b[2],y2 = b[4])
                    for b in boxes
                ],
                shape = img_r.shape
            ).on(img).to_xyxy_array()
            
        else:
            boxes  = tracker.update(img)
            
        n      = len(boxes)
        if n  > 0:
            if not logname is None:
                log   = metric.convert_motchallenge(frame_id,boxes)
                for l in log:
                    print(l,file = f)
            img    = visualize_det_cv2_part(img,scores=None,
                                            cls_ids=[0] * len(boxes),
                                            boxes = boxes[:,1:],
                                            track_ids=boxes[:,0].astype("int").tolist(),
                                            class_names=class_names,
                                            thresh = -1
                                        )
        frame_id += 1
        frames.append(img)
    f.close()
    
    clip = ImageSequenceClip(frames,fps = fps).resize(scale)
    if not gif:
        clip.write_videofile(output_video)
    else:
        clip.write_gif(output_video)