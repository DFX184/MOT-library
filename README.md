# MOT-library


multiple object tracking 

## Demo

### Demo 1

https://user-images.githubusercontent.com/59595277/185565011-d46455e9-5022-477a-abe4-45443e141362.mp4

### Demo 2


https://user-images.githubusercontent.com/59595277/185734746-862c86e6-e5fe-4028-979f-9f8159a1c9a7.mp4



### Demo 3


https://user-images.githubusercontent.com/59595277/185734752-511c643b-90de-4eec-9339-743a4b12f463.mp4


## Usage

```py
from byte_mot import *
detector = xxx # your detector
mot      = ByteMot(detecher)
mot.update(img) # where image is numpy arrary
```

```py
from detector import YoloX
yolox = YoloX.YoloXInference("bytetrack_x_mot17")
yolox(img) #
```
