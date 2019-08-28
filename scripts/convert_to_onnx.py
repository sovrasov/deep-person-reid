import torch
import numpy as np
from PIL import Image

from deprecated.default_parser import init_parser
from torchreid.data.transforms import build_transforms
import torchreid
from torchreid.utils import load_pretrained_weights




parser = init_parser()
parser.add_argument('--output_name', type=str, default='model')
args = parser.parse_args()

model = torchreid.models.build_model(
        name=args.arch,
        num_classes=1041,
        loss=args.loss.lower(),
        pretrained=(not args.no_pretrained),
        use_gpu=True,
        feature_dim=args.feature_dim
    )
load_pretrained_weights(model, args.load_weights)
model.eval()

_, transform = build_transforms(args.height, args.width)

input_size = (args.height, args.width, 3)
img = np.random.rand(*input_size).astype(np.float32)
img = np.uint8(img*255)
im = Image.fromarray(img)
blob = transform(im).unsqueeze(0)

torch.onnx.export(model, blob, args.output_name + '.onnx',
                  verbose=True, export_params=True)
