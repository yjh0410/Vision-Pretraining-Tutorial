from .mlp.build import build_mlp
from .convnet.build import build_convnet


def build_model(args):
    # --------------------------- ResNet series ---------------------------
    if 'mlp' in args.model:
        model = build_mlp(args)
    elif 'convnet' in args.model:
        model = build_convnet(args)
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))

    return model
