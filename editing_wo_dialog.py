import argparse
import logging
import os
import numpy as np
import torch

from models import create_model
from models.utils import save_image
from utils.editing_utils import edit_target_attribute
from utils.inversion_utils import inversion
from utils.logger import get_root_logger
from utils.options import dict2str, dict_to_nonedict, parse, parse_opt_wrt_resolution
from utils.util import make_exp_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Talk-to-Edit (CPU Safe Version)')
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--attr', type=str, required=True, help='Attribute to edit')
    parser.add_argument('--target_val', type=int, required=True, help='Target value')
    parser.add_argument(
        '--latent_index',
        type=int,
        default=None,
        help='Optional latent face index from teaser dataset (0-99).'
    )
    parser.add_argument(
        '--input_image',
        type=str,
        default=None,
        help='Optional path to a real input image for inversion-based editing.'
    )
    parser.add_argument(
        '--preview_only',
        action='store_true',
        help='Only synthesize and save start image for preview, then exit.'
    )
    parser.add_argument(
        '--preview_output',
        type=str,
        default=None,
        help='Optional output path for preview image.'
    )
    return parser.parse_args()


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    return device


def load_or_generate_latent(opt, field_model, device):
    if opt['inversion']['is_real_image']:
        latent_code = inversion(opt, field_model)
        return latent_code

    if opt['latent_code_path'] is None:
        print("[INFO] Generating random latent code...")
        latent = torch.randn(1, 512, device=device)
        with torch.no_grad():
            latent = field_model.stylegan_gen.get_latent(latent)
        latent = latent.cpu().numpy()

        np.save(f'{opt["path"]["visualization"]}/latent_code.npy', latent)
        return latent

    print("[INFO] Loading latent code from file...")
    i = opt['latent_code_index']
    latent = np.load(opt['latent_code_path'], allow_pickle=True).item()[f"{str(i).zfill(7)}.png"]

    latent = torch.from_numpy(latent).to(device)
    with torch.no_grad():
        latent = field_model.stylegan_gen.get_latent(latent)

    return latent.cpu().numpy()


def main():
    args = parse_args()
    device = get_device()

    # ---------- Load config ----------
    opt = parse(args.opt, is_train=False)
    if args.latent_index is not None:
        opt['latent_code_index'] = int(args.latent_index)
        print(f"[INFO] Using latent index: {args.latent_index}")
    if args.input_image:
        opt['inversion']['is_real_image'] = True
        opt['inversion']['img_path'] = args.input_image
        print(f"[INFO] Using input image: {args.input_image}")
    opt = parse_opt_wrt_resolution(opt)
    make_exp_dirs(opt)
    opt = dict_to_nonedict(opt)

    # ---------- Logger ----------
    log_path = opt["path"]["log"]
    os.makedirs(log_path, exist_ok=True)

    logger = get_root_logger(
        logger_name='editing',
        log_level=logging.INFO,
        log_file=f'{log_path}/editing.log'
    )
    logger.info(dict2str(opt))

    vis_path = opt["path"]["visualization"]
    os.makedirs(vis_path, exist_ok=True)

    # ---------- Model ----------
    field_model = create_model(opt)

    # ---------- Latent ----------
    latent_code = load_or_generate_latent(opt, field_model, device)

    # ---------- Synthesize ----------
    with torch.no_grad():
        latent_tensor = torch.from_numpy(latent_code).to(device)
        start_image, start_label, start_score = field_model.synthesize_and_predict(latent_tensor)

    save_image(start_image, f'{vis_path}/start_image.png')
    if args.preview_output:
        save_image(start_image, args.preview_output)

    if args.preview_only:
        print("[INFO] Preview image generated. Skipping edit step.")
        return

    # ---------- Attributes ----------
    attribute_dict = {
        "Bangs": start_label[0],
        "Eyeglasses": start_label[1],
        "No_Beard": start_label[2],
        "Smiling": start_label[3],
        "Young": start_label[4],
    }

    edit_label = {
        'attribute': args.attr,
        'target_score': args.target_val
    }

    edited_latent_code = None
    round_idx = 0

    print("[INFO] Starting attribute editing...")

    attribute_dict, exception_mode, latent_code, edited_latent_code = edit_target_attribute(
        opt,
        attribute_dict,
        edit_label,
        round_idx,
        latent_code,
        edited_latent_code,
        field_model,
        logger,
        print_intermediate_result=True
    )

    # ---------- Handle exceptions ----------
    if exception_mode != 'normal':
        if exception_mode == 'already_at_target_class':
            logger.info("Already at desired attribute level.")
        elif exception_mode == 'max_edit_num_reached':
            logger.info("Editing limit reached. Try different attribute.")

    print("[INFO] Done.")


if __name__ == '__main__':
    main()
