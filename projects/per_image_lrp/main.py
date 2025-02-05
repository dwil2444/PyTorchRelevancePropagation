"""Per-image Layer-wise Relevance Propagation

This script uses a pre-trained VGG network from PyTorch's Model Zoo
to perform Layer-wise Relevance Propagation (LRP) on the images
stored in the 'input' folder.

NOTE: LRP supports arbitrary batch size. Plot function does currently support only batch_size=1.

"""
import argparse
import time
import pathlib

from logger.custom_logger import CustomLogger
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.models as models

from torch_lrp.data import get_data_loader
from torch_lrp.lrp import LRPModel

from projects.per_image_lrp.visualize import plot_relevance_scores
logger = CustomLogger(__name__).logger


def per_image_lrp(config: argparse.Namespace) -> None:
    """Test function that plots heatmaps for images placed in the input folder.

    Images have to be placed in their corresponding class folders.

    Args:
        config: Argparse namespace object.

    """
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logger.info(f'Using Device: {device}')
    data_loader = get_data_loader(config)

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.to(device)


    

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        sm = nn.Softmax(dim=1)
        # y = y.to(device)  # here not used as method is unsupervised.
        # get model confidence for target label
        t0 = time.time()
        eps = sm(model(x))
        logger.info(torch.sigmoid(torch.max(eps)).item())
        exit()
        lrp_model = LRPModel(model=model, 
                             top_k=config.top_k,
                             eps=eps)
        synth_relevance = torch.randn(1, 2048, 19, 29).to(device)
        r = lrp_model.forward(x,
                              synth_relevance=synth_relevance)
        logger.info("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))
        plot_relevance_scores(x=x, r=r, name=str(i), config=config)
        break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input directory.",
        default="./input/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory.",
        default="./output/",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", help="Batch size.", default=1, type=int
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device.",
        choices=["gpu", "cpu"],
        default="gpu",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        help="Proportion of relevance scores that are allowed to pass.",
        default=0.02,
        type=float,
    )
    parser.add_argument(
        "-r",
        "--resize",
        dest="resize",
        help="Resize image before processing.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        dest="epsilon",
        help="Epsilon value for relevance propagation",
        default=0.0,
        type=float,
    )

    config = parser.parse_args()

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    per_image_lrp(config=config)
