import argparse
import cv2
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import datetime
# d = datetime.datetime.now()

def restoration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--arch', type=str, default='clean')
    parser.add_argument('--channel', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan')
    parser.add_argument('--bg_tile', type=int, default=0)
    parser.add_argument('--test_path', type=str, default='static')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--paste_back', action='store_false')
    parser.add_argument('--save_root', type=str, default='static')

    args = parser.parse_args()
    if args.test_path.endswith('/'):
        args.test_path = args.test_path[:-1]
    os.makedirs(args.save_root, exist_ok=True)

    # background upsampler
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            bg_upsampler = None
        else:
            from realesrgan import RealESRGANer
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=False)  # need to set True in GPU mode
    else:
        bg_upsampler = None
    # set up GFPGAN restorer
    restorer = GFPGANer(
        model_path=args.model_path,
        upscale=args.upscale,
        arch=args.arch,
        channel_multiplier=args.channel,
        bg_upsampler=bg_upsampler)

    img_path = 'static/input.png'
    # read image
    img_name = os.path.basename(img_path)
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    _, restored_face, restored_img = restorer.enhance(
            input_img, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=args.paste_back)

    # save restored face
    d = datetime.datetime.now()
    save_face_name = 'face.png'
    save_restore_path = os.path.join(args.save_root, save_face_name)
    imwrite(restored_face[0], save_restore_path)
    imwrite(restored_face[0], f'outputs/face_{d.second}.png')
    # save restored img
    save_restore_path = os.path.join(args.save_root, 'output.png')
    imwrite(restored_img, save_restore_path)
    imwrite(restored_img, f'outputs/image_{d.second}.png')
