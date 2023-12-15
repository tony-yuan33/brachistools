
import argparse
from pathlib import Path
import os, sys

try:
    from brachistools.gui import gui
    GUI_ENABLED = True
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = False
    raise

import logging

from brachistools.segmentation import segmentation_pipeline, default_segmentation_params
# TODO: Import components
# from brachistools.classification import ...
from brachistools.io import load_folder
from brachistools.version import version_str

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Brachistools Command Line Parameters")

    parser.add_argument('--version', action='store_true', help="show brachistools version info")
    parser.add_argument('--verbose', action='store_true', help="print additional messages")

    parser.add_argument('command', type=str, choices=['segment', 'classify', 'config', 'gui'])

    subparsers = parser.add_subparsers(dest='command')

    segment_subparser = subparsers.add_parser('segment', help="perform segmentation")
    classify_subparser = subparsers.add_parser('classify', help="perform classification (suggest diagnosis)")
    config_subparser = subparsers.add_parser('config', help="program configurations")

    input_img_args = parser.add_argument_group("Input Image Arguments")
    input_img_args.add_argument('--dir', default=[], type=str, help='folder containing data to run on')
    input_img_args.add_argument('--image_path', default=[], type=str,
                                help='run on single image')

    segmentation_args = segment_subparser.add_argument_group("Segmentation Pipeline Arguments")
    segmentation_args.add_argument('--vahadane-sparsity_regularizer',
                                   required=False, default=0.75, type=float,
                                   help="sparsity regularizer of dictionary learning in Vahadane's "
                                   "H&E deconvolution algorithm. Smaller values lead to less learning "
                                   "capability and usually result in more complete nucleus shapes (increases "
                                   "false positive rate)")
    segmentation_args.add_argument('--equalize_adapthist-clip_limit',
                                   required=False, default=0.01, type=float,
                                   help="clip_limit parameter in skimage.exposure.equalize_adapthist")
    segmentation_args.add_argument('--small_objects-min_size',
                                   required=False, default=250, type=int,
                                   help="minimum threshold size of connected regions of 1")
    segmentation_args.add_argument('--small_holes-area_threshold',
                                   required=False, default=100, type=int,
                                   help="maximum threshold size of connected regions of 0")
    segmentation_args.add_argument('--local_max-min_distance',
                                   required=False, default=12, type=int,
                                   help="minimum distance between two local maxima. Combats over-segmentation")
    segmentation_args.add_argument('--local_max-threshold_rel',
                                   required=False, default=0.2, type=float,
                                   help="filter smaller maxima based on (this value * max(all_maxima)). "
                                   "Combats over-segmentation")
    segmentation_args.add_argument('--small_labels-min_size',
                                   required=False, default=300, type=int,
                                   help="minimum size of an independent label; smaller labels will "
                                   "be merged to their largest neighbors. Combats over-segmentation")

    # TODO: Add args for classification
    # classification_args = classify_subparser.add_argument_group("Classification Algorithm Arguments")
    # classification_args.add_argument(...)

    # TODO: Add args for hardware
    # hardware_args = classify_subparser.add_argument_group("Hardware Arguments")
    # hardware_args.add_argument('--use_gpu', action='store_true', help='use GPU if tensorflow with CUDA installed')
    # hardware_args.add_argument('--gpu_device', required=False, default='0', type=str,
    #                            help='which GPU device to use, use an integer for CUDA, or mps for M1')

    config_subparser.add_argument('--param_dir', required=False, default='models', type=str,
                                  help="folder of model parameters")

    output_args = parser.add_argument_group("Output Arguments")
    output_args.add_argument('--save_format', required=False, default='PNG', type=str,
                             help="the file extension (no dot) of saved binary masks. Default is 'PNG'")
    output_args.add_argument('--save_dir', default=None, type=str,
                             help="folder to which segmentation results will be saved (defaults to input image directory)")
    output_args.add_argument('--save_npy', action='store_true',
                             help="save instance segmentation results as '.npy' labeled mask arrays. "
                             "The XML format for instance segmentation will always be saved regardless of "
                             "this option")
    return parser

def main():
    args = get_arg_parser().parse_args()

    if args.version:
        print(version_str)
        return

    if args.verbose:
        from brachistools.io import logger_setup
        logger, _ = logger_setup()
    else:
        logger = logging.getLogger(__name__)

    # TODO: Assign devices
    # from brachistools import classification
    # device, gpu = classification.assign_device(use_tensorflow=True, gpu=args.use_gpu, device=...)

    if args.command == 'gui':
        if not GUI_ENABLED:
            print('GUI ERROR:', GUI_ERROR)
            if GUI_IMPORT:
                print('GUI FAILED: GUI dependencies may not be installed, to install, run')
                print('     pip install "brachistools[gui]"')
        else:
            gui.run()

    if args.command == 'config':
        from configparser import ConfigParser

        try:
            config_path = Path.home().joinpath(
                '.brachistools', 'config.ini')
            config = ConfigParser()
            config.read(config_path)
        except:
            logger.critical("Failed to open config file")
            sys.exit(-1)

        config.set('ModelParams', 'param_dir', args.param_dir)

        with open(config_path, 'w') as config_f:
            config.write(config_f)

    # Prepare images
    if args.dir and args.image_path:
        logger.critical("Cannot specify both --dir and --image_path")
        sys.exit(-1)

    if args.dir:
        image_names = load_folder(args.dir)
    elif args.image_path:
        image_names = [args.image_path]

    if args.command == 'segment':
        ...

    if args.command == 'classify':
        ...
