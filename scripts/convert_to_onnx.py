import sys
import os
import os.path as osp
import warnings
import time

import torch
import torch.nn as nn

from default_parser import (
    init_parser, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
from torchreid.data.transforms import build_transforms
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)


import glog as log
import numpy as np
import cv2 as cv


parser = init_parser()
args = parser.parse_args()

model = torchreid.models.build_model(
        name=args.arch,
        num_classes=1041,
        loss=args.loss.lower(),
        pretrained=(not args.no_pretrained),
        use_gpu=True
    )
load_pretrained_weights(model, args.load_weights)
model.eval()

_, transform = build_transforms(args.height, args.width)

input_size=(args.height, args.width, 3)
img = np.random.rand(*input_size).astype(np.float32)
img = np.uint8(img*255)
from PIL import Image
im = Image.fromarray(img)
blob = transform(im).unsqueeze(0)


torch.onnx.export(model, blob, 'model_3_1' + '.onnx', verbose=True, export_params=True)
