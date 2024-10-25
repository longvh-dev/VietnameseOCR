import os
import cv2

from paddleocr import PPStructure,draw_structure_result,save_structure_res
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cuda'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False

table_engine = PPStructure(lang='en', use_vietocr=True, vietocr_config=config)

save_folder = './output'
img_path = '' # path to your image
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'font/ARIAL.TTF' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')