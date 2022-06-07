import torch
import argparse


def create_base_parser():
    """Define the common options that are used in preprocessing."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', required=True, help='path to input images')
    parser.add_argument('--output_dir', required=False, type=str, default='', help='path to output images')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--tile_size', required=False, type=int, default=512, help='size of the tiles')
    parser.add_argument('--overlap_size', required=False, type=int, default=50,
                        help='size of the overlapping area between tiles')
    parser.add_argument('--gpu_ids', required=False, type=str, default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epoch_count', type=int, default=0,
                             help='the starting  epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

    return parser


class ProcessingOptions():
    """This class defines options used during preprocessing."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized."""
        self.parser = create_base_parser()

    def gather_options(self):
        """Initialize preprocessing parser with basic options."""
        return self.parser.parse_args()

    def print_options(self, opt):
        """Print options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        return opt
