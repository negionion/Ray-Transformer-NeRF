from .models import PixelNeRFNet
from torchsummary import summary

def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    summary(net, [(1, 42, 1), (3, 64, 64)])
    return net
