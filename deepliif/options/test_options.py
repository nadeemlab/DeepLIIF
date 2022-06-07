from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self):
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        self.parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        self.parser.add_argument('--num_test', type=int, default=10000, help='how many test images to run')
        # rewrite devalue values
        self.parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        self.parser.set_defaults(load_size=self.parser.get_default('crop_size'))
        self.is_train = False
