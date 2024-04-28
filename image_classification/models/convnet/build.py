from .convnet import ConvNet

def build_convnet(args):
    if args.model == "convnet":
        model = ConvNet(img_size      = args.img_size,
                        in_dim        = args.img_dim,
                        hidden_dim    = 16,
                        num_classes   = args.num_classes,
                        act_type      = "sigmoid",
                        norm_type     = "bn",
                        use_adavgpool = True)
        
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))
    
    return model
