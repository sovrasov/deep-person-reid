import argparse

import torch
import numpy as np
from PIL import Image

from torchreid.data.transforms import build_transforms
import torchreid
from torchreid.utils import load_pretrained_weights

from default_config import get_default_config


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--output_name', type=str, default='model')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=2,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        dropout_prob=cfg.model.dropout_prob,
        feature_dim=cfg.model.feature_dim,
        activation=cfg.model.activation,
        in_first=cfg.model.in_first
        )
    load_pretrained_weights(model, cfg.model.load_weights)
    model.eval()

    _, transform = build_transforms(cfg.data.height, cfg.data.width, cfg.data.transforms)

    input_size = (cfg.data.height, cfg.data.width, 3)
    img = np.random.rand(*input_size).astype(np.float32)
    img = np.uint8(img*255)
    im = Image.fromarray(img)
    blob = transform(im).unsqueeze(0)

    torch.onnx.export(model, blob, args.output_name + '.onnx',
                      verbose=True, export_params=True)


if __name__ == '__main__':
    main()
