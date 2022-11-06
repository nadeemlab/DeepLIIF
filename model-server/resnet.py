from deepliif.models import ResnetGenerator, get_norm_layer


class Resnet(ResnetGenerator):
    def __init__(self):
        super(Resnet, self).__init__(
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm_layer=get_norm_layer(norm_type='batch'),
            use_dropout=True,
            n_blocks=9,
            padding_type='zero'
        )
