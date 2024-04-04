from .convnet import ConvNet

def build_convnet(args):
    if args.model == "convnet":
        model = ConvNet(in_dim     = args.img_dim,
                        inter_dim  = 256,
                        out_dim    = args.num_classes,
                        act_type   = "sigmoid",
                        norm_type  = "bn",
                        avgpool=True)
        
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))
    
    return model
