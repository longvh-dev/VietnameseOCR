import yaml
from vietocr.tool.utils import download_config

url_config = {
        'vgg_transformer':'vietocr/config/vgg-transformer.yml',
        'resnet_transformer':'vietocr/config/resnet_transformer.yml',
        'resnet_fpn_transformer':'vietocr/config/resnet_fpn_transformer.yml',
        'vgg_seq2seq':'vietocr/config/vgg-seq2seq.yml',
        'vgg_convseq2seq':'vietocr/config/vgg_convseq2seq.yml',
        'vgg_decoderseq2seq':'vietocr/config/vgg_decoderseq2seq.yml',
        'base':'vietocr/config/base.yml',
        }

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        #base_config = download_config(url_config['base'])
        base_config = {}
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config['base'])
        config = download_config(url_config[name])

        base_config.update(config)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

