import sys
if './' not in sys.path:
	sys.path.append('./')
from utils.share import *
import utils.config as config

import os
import cv2
import torch
import einops
import argparse
import numpy as np

from PIL import Image
from datasets import load_dataset
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector
from annotator.content import ContentDetector

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler


apply_canny = CannyDetector()
apply_mlsd = MLSDdetector()
apply_hed = HEDdetector()
apply_sketch = SketchDetector()
apply_openpose = OpenposeDetector()
apply_midas = MidasDetector()
apply_seg = UniformerDetector()
apply_content = ContentDetector()


model = create_model('./configs/uni_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./ckpt/uni.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(canny_image, mlsd_image, hed_image, sketch_image, openpose_image, midas_image, seg_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):

    seed_everything(seed)

    if canny_image is not None:
        anchor_image = canny_image
    elif mlsd_image is not None:
        anchor_image = mlsd_image
    elif hed_image is not None:
        anchor_image = hed_image
    elif sketch_image is not None:
        anchor_image = sketch_image
    elif openpose_image is not None:
        anchor_image = openpose_image
    elif midas_image is not None:
        anchor_image = midas_image
    elif seg_image is not None:
        anchor_image = seg_image
    elif content_image is not None:
        anchor_image = content_image
    else:
        anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape

    with torch.no_grad():
        # Canny
        if canny_image is not None:
            canny_image = cv2.resize(canny_image, (W, H))
            canny_detected_map = HWC3(apply_canny(HWC3(canny_image), low_threshold, high_threshold))
        else:
            canny_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # MLSD
        if mlsd_image is not None:
            mlsd_image = cv2.resize(mlsd_image, (W, H))
            mlsd_detected_map = HWC3(apply_mlsd(HWC3(mlsd_image), value_threshold, distance_threshold))
        else:
            mlsd_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # Hed
        if hed_image is not None:
            hed_image = cv2.resize(hed_image, (W, H))
            hed_detected_map = HWC3(apply_hed(HWC3(hed_image)))
        else:
            hed_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # Sketch
        if sketch_image is not None:
            sketch_image = cv2.resize(sketch_image, (W, H))
            sketch_detected_map = HWC3(apply_sketch(HWC3(sketch_image)))
        else:
            sketch_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # Pose
        if openpose_image is not None:
            openpose_image = cv2.resize(openpose_image, (W, H))
            openpose_detected_map = openpose_image

            # openpose_detected_map, _ = apply_openpose(HWC3(openpose_image), False)
            # openpose_detected_map = HWC3(openpose_detected_map)
        else:
            openpose_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # Midas Depth
        if midas_image is not None:
            midas_image = cv2.resize(midas_image, (W, H))
            midas_detected_map = HWC3(apply_midas(HWC3(midas_image), alpha))
        else:
            midas_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # Segmentation
        if seg_image is not None:
            seg_image = cv2.resize(seg_image, (W, H))
            seg_detected_map = seg_image

            # seg_detected_map, _ = apply_seg(HWC3(seg_image))
            # seg_detected_map = HWC3(seg_detected_map)
        else:
            seg_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        # Content
        if content_image is not None:
            content_emb = apply_content(content_image)
        else:
            content_emb = np.zeros((768))

        detected_maps_list = [canny_detected_map,
                              mlsd_detected_map,
                              hed_detected_map,
                              sketch_detected_map,
                              openpose_detected_map,
                              midas_detected_map,
                              seg_detected_map
                              ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)

        cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, global_strength=global_strength)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate edge tasks")
    parser.add_argument('--task', type=str, default='canny')
    parser.add_argument("--dataset_name", type=str, default='limingcv/MultiGen-20M_canny_eval')
    parser.add_argument("--cache_dir", type=str, default='data/huggingface_datasets')
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--strength", type=float, default=1.0, help='control guidiance strength')
    parser.add_argument("--guidiance_scale", type=float, default=7.5, help='text guidiance scale')
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--prompt_column", type=str, default='text')
    parser.add_argument("--condition_column", type=str, default='image')
    parser.add_argument("--image_resolution", type=int, default=512)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir, split=args.split)

    for idx, data in enumerate(dataset):
        print(f'Processing {idx}/{len(dataset)} image')
        prompt = data[args.prompt_column]
        condition = data[args.condition_column].resize((512, 512))
        condition = condition.convert('RGB')
        condition = np.array(condition)[:, :, ::-1].copy()  # PIL to CV2 format

        canny_image=None
        mlsd_image=None
        hed_image=None
        sketch_image=None
        openpose_image=None
        midas_image=None
        seg_image=None
        low_threshold = high_threshold = 0

        save_path = os.path.join('outputs', args.task, args.dataset_name, args.split)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(os.path.join(save_path, 'annotations'))
            for i in range(args.num_samples):
                os.makedirs(f'{save_path}/group_{i}')

        if args.task == 'canny':
            canny_image = condition

            low_threshold = np.random.randint(0, 255)
            high_threshold = np.random.randint(low_threshold, 255)
            annotation = apply_canny(np.array(condition), low_threshold, high_threshold)
            annotation = Image.fromarray(np.uint8(annotation)).convert('RGB')
            annotation.save(f'{save_path}/annotations/{idx}.png')
        elif args.task =='depth':
            midas_image = condition

            annotation = apply_midas(np.array(condition), 6.2)
            annotation = Image.fromarray(np.uint8(annotation)).convert('RGB')
            annotation.save(f'{save_path}/annotations/{idx}.png')
        elif args.task == 'hed':
            hed_image = condition

            annotation = apply_hed(np.array(condition))
            annotation = Image.fromarray(np.uint8(annotation)).convert('RGB')
            annotation.save(f'{save_path}/annotations/{idx}.png')
        elif args.task == 'openpose':
            openpose_image = condition
            annotation = Image.fromarray(np.uint8(condition)).convert('RGB')
            annotation.save(f'{save_path}/annotations/{idx}.png')
        elif args.task =='seg':
            seg_image = data[args.condition_column].resize((512, 512))
            seg_image = np.array(seg_image)

            annotation = data[args.condition_column].resize((512, 512))
            annotation.save(f'{save_path}/annotations/{idx}.png')

        images = process(
            canny_image=canny_image,
            mlsd_image=mlsd_image,
            hed_image=hed_image,
            sketch_image=sketch_image,
            openpose_image=openpose_image,
            midas_image=midas_image,
            seg_image=seg_image,
            content_image=None,
            prompt=prompt,
            a_prompt='best quality, extremely detailed',
            n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
            num_samples=args.num_samples,
            image_resolution=args.image_resolution,
            ddim_steps=args.ddim_steps,
            strength=args.strength,
            scale=args.guidiance_scale,
            seed=42,
            eta=0.0,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            value_threshold=0.1,
            distance_threshold=0.1,
            alpha=6.2,
            global_strength=1.0,
        )

        for i, img in enumerate(images):
            Image.fromarray(np.uint8(img)).save(f'{save_path}/group_{i}/{idx}.png')

            if args.task == 'canny':
                canny_image = apply_canny(img, low_threshold, high_threshold)
                Image.fromarray(np.uint8(canny_image)).save(f'{save_path}/group_{i}/{idx}_canny.png')
            elif args.task == 'hed':
                hed_image = apply_hed(img)
                Image.fromarray(np.uint8(hed_image)).save(f'{save_path}/group_{i}/{idx}_hed.png')
            elif args.task == 'depth':
                midas_image = apply_midas(img, 6.2)
                Image.fromarray(np.uint8(midas_image)).save(f'{save_path}/group_{i}/{idx}_depth.png')


# CUDA_VISIBLE_DEVICES=0 python3 eval.py --task='hed' --dataset_name='limingcv/MultiGen-20M_canny_eval' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='text' --condition_column='image' --image_resolution=512

# CUDA_VISIBLE_DEVICES=1 python3 eval.py --task='canny' --dataset_name='limingcv/MultiGen-20M_canny_eval' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='text' --condition_column='image' --image_resolution=512

# CUDA_VISIBLE_DEVICES=0 python3 eval.py --task='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='text' --condition_column='image' --image_resolution=512

# CUDA_VISIBLE_DEVICES=1 python3 eval.py --task='openpose' --dataset_name='limingcv/HumanArtv2' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='prompt' --condition_column='control_pose' --image_resolution=512

# CUDA_VISIBLE_DEVICES=0 python3 eval.py --task='openpose' --dataset_name='limingcv/Captioned_COCOPose' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='prompt' --condition_column='control_pose' --image_resolution=512

# CUDA_VISIBLE_DEVICES=1 python3 eval.py --task='seg' --dataset_name='limingcv/Captioned_ADE20K' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='prompt' --condition_column='control_seg' --image_resolution=512