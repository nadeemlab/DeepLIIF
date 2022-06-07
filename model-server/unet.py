from deepliif.models import UnetGenerator, get_norm_layer


class Unet(UnetGenerator):
    def __init__(self):
        super(Unet, self).__init__(
            input_nc=3,
            output_nc=3,
            num_downs=9,
            ngf=64,
            norm_layer=get_norm_layer(norm_type='batch'),
            use_dropout=True,
        )
