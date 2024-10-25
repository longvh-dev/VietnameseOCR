# PaddleOCR + VietOCR: Công cụ OCR Tiếng Việt tuyệt vời

# Sử dụng 
Thêm tham số `use_vietocr=True` và khởi tạo config cho VietOCR
```
import os
import cv2
from PIL import Image
from paddleocr import PPStructure, save_structure_res, PaddleOCR, draw_ocr
from vietocr.tool.config import Cfg

# Khởi tạo VietOCR config
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cuda'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False

# OCR
ocr = PaddleOCR(lang='en', use_vietocr=True, vietocr_config=config)

img_path = '' # path to your image
img = cv2.imread(img_path)

result = ocr.ocr(img)

# Lưu kết quả
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='font/ARIAL.TTF')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

# Document 
Tham khảo [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) và [VietOCR](https://github.com/pbcquoc/vietocr)
