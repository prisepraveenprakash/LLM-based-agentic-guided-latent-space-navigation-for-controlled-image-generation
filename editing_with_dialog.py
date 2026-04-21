import argparse
import json
import logging
import os.path

import numpy as np
import torch

from models import create_model
from utils.dialog_edit_utils import dialog_with_real_user
from utils.inversion_utils import inversion
from utils.logger import get_root_logger
from utils.options import (dict2str, dict_to_nonedict, parse,
                           parse_args_from_opt, parse_opt_wrt_resolution)
from utils.util import make_exp_dirs


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--opt', default=None, type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--request',
        action='append',
        default=None,
        help='Optional request text. Use multiple times for multi-turn scripted dialog.'
    )
    parser.add_argument(
        '--request_file',
        default=None,
        type=str,
        help='Optional JSON file containing a list of dialog requests.'
    )
    parser.add_argument(
        '--auto_end_on_exhausted',
        action='store_true',
        help='When scripted requests are exhausted, auto-send a fallback ending request.'
    )
    parser.add_argument(
        '--fallback_request_text',
        default="That's all, thank you.",
        type=str,
        help='Fallback user request used when scripted requests are exhausted.'
    )
    parser.add_argument(
        '--max_rounds',
        default=None,
        type=int,
        help='Optional maximum dialog rounds for non-interactive usage.'
    )
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
    return parser.parse_args()


def main():

    # ---------- Set up -----------
    args = parse_args()
    opt = parse(args.opt, is_train=False)
    if args.latent_index is not None:
        opt['latent_code_index'] = int(args.latent_index)
    if args.input_image:
        opt['inversion']['is_real_image'] = True
        opt['inversion']['img_path'] = args.input_image
    opt = parse_opt_wrt_resolution(opt)
    args = parse_args_from_opt(args, opt)
    make_exp_dirs(opt)

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # set up logger
    save_log_path = f'{opt["path"]["log"]}'
    dialog_logger = get_root_logger(
        logger_name='dialog',
        log_level=logging.INFO,
        log_file=f'{save_log_path}/dialog.log')
    dialog_logger.info(dict2str(opt))

    save_image_path = f'{opt["path"]["visualization"]}'
    os.makedirs(save_image_path, exist_ok=True)

    scripted_requests = []
    if args.request_file:
        with open(args.request_file, 'r') as f:
            loaded_requests = json.load(f)
            if not isinstance(loaded_requests, list):
                raise ValueError('request_file must contain a JSON list.')
            scripted_requests.extend([str(item) for item in loaded_requests])
    if args.request:
        scripted_requests.extend([str(item) for item in args.request])
    if scripted_requests:
        args.request_queue = scripted_requests
        if not args.auto_end_on_exhausted:
            args.auto_end_on_exhausted = True

    # ---------- Load files -----------
    dialog_logger.info('loading template files')
    with open(opt['feedback_templates_file'], 'r') as f:
        args.feedback_templates = json.load(f)
        args.feedback_replacement = args.feedback_templates['replacement']
    with open(opt['pool_file'], 'r') as f:
        pool = json.load(f)
        args.synonyms_dict = pool["synonyms"]

    # ---------- create model ----------
    field_model = create_model(opt)
    device = field_model.device

    # ---------- load latent code ----------
    if opt['inversion']['is_real_image']:
        latent_code = inversion(opt, field_model)
    else:
        if opt['latent_code_path'] is None:
            latent_code = torch.randn(1, 512, device=device)
            with torch.no_grad():
                latent_code = field_model.stylegan_gen.get_latent(latent_code)
            latent_code = latent_code.cpu().numpy()
            np.save(f'{opt["path"]["visualization"]}/latent_code.npz.npy',
                    latent_code)
        else:
            i = opt['latent_code_index']
            latent_code = np.load(
                opt['latent_code_path'],
                allow_pickle=True).item()[f"{str(i).zfill(7)}.png"]
            latent_code = torch.from_numpy(latent_code).to(device)
            with torch.no_grad():
                latent_code = field_model.stylegan_gen.get_latent(latent_code)
            latent_code = latent_code.cpu().numpy()

    np.save(f'{opt["path"]["visualization"]}/latent_code.npz.npy', latent_code)

    # ---------- Perform dialog-based editing with user -----------
    dialog_overall_log = dialog_with_real_user(field_model, latent_code, opt,
                                               args, dialog_logger)

    # ---------- Log the dialog history -----------
    for (key, value) in dialog_overall_log.items():
        dialog_logger.info(f'{key}: {value}')
    dialog_logger.info('successfully end.')


if __name__ == '__main__':
    main()
