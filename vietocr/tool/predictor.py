from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict
from vietocr.tool.utils import download_weights

import torch
import numpy as np
from typing import List, Union
from collections import defaultdict
import time

class Predictor():
    def __init__(self, config):

        device = config['device']
        
        model, vocab = build_model(config)
        weights = '/tmp/weights.pth'

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device

    # def predict(self, 
    #             img: np.ndarray, 
    #             return_prob: bool = False):
    #     img = process_input(img, self.config['dataset']['image_height'], 
    #             self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
    #     img = img.to(self.config['device'])

    #     if self.config['predictor']['beamsearch']:
    #         sent = translate_beam_search(img, self.model)
    #         s = sent
    #         prob = None
    #     else:
    #         s, prob = translate(img, self.model)
    #         s = s[0].tolist()
    #         prob = prob[0]

    #     s = self.vocab.decode(s)
        
    #     if return_prob:
    #         return s, prob
    #     else:
    #         return s

    def __call__(self, 
                    imgs: Union[np.ndarray, List[np.ndarray]]
                    ):
        st = time.time()

        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}

        if not isinstance(imgs, list):
            imgs = [imgs]    

        rec_res = [["", 0.0]] * len(imgs)

        for i, img in enumerate(imgs):
            img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        
            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)


        for k, batch in bucket.items():
            batch = torch.cat(batch, 0).to(self.device)
            s, prob = translate(batch, self.model)
            prob = prob.tolist()

            s = s.tolist()
            s = self.vocab.batch_decode(s)

            bucket_pred[k] = (s, prob)


        for k in bucket_pred:
            idx = bucket_idx[k]
            sent, prob = bucket_pred[k]
            for i, j in enumerate(idx):
                rec_res[j] = (sent[i], prob[i])

        return rec_res, time.time()-st


