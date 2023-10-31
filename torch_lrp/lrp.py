"""Class for layer-wise relevance propagation.

Layer-wise relevance propagation for VGG-like networks from PyTorch's Model Zoo.
Implementation can be adapted to work with other architectures as well by adding the corresponding operations.

    Typical usage example:

        model = torchvision.models.vgg16(pretrained=True)
        lrp_model = LRPModel(model)
        r = lrp_model.forward(x)

"""
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Optional, List, Tuple
from typing import OrderedDict as TypingOrderedDict
OrderedDictType = TypingOrderedDict[str, nn.Module]
from torch_lrp.utils import layers_lookup
from logger.custom_logger import CustomLogger
logger = CustomLogger(__name__).logger


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module, 
                 eps: float=1.0e-05,
                 top_k: float = 0.0) -> None:
        """
        Args:

            model:
            eps:
            top_k:
        """
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.eps = eps # remember to add epsilon function
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!
        self.all_layer_dict: OrderedDictType = OrderedDict()
        self.all_layer_names = self._get_modules(module=self.model)
        self.operational_layers = self._get_layer_operations()
        self.no_match = {} # keep track of mystery layers
        # Create LRP network
        self.lrp_layers = self._create_lrp_model()
        self.preceding_layers = []


    def get_last_conv(self) -> Tuple[str, nn.Module]:
        """
        """
        reversed_layers = reversed(self.operational_layers)
        for layer in reversed_layers:
            if isinstance(layer,  nn.Conv2d):
                return (layer.module_name, layer)
            else:
                self.preceding_layers.append(layer)


    def _get_modules(self, module: nn.Module,
                                parent_name: str ="model") -> List[str]:
        """
        """
        module_names = []
        if parent_name == 'model':
            current_name = module._get_name()
            setattr(module, 'module_name', current_name)
            #module_names.append(current_name)
        for name, child in module.named_children():
            if parent_name == "":
                current_name = name
            else:
                current_name = f"{parent_name}.{name}"
            setattr(child, 'module_name', current_name)
            self.all_layer_dict[current_name] = child
            module_names.append(current_name)
            module_names.extend(self._get_modules(child, current_name))
        return module_names
    

    def _get_layer_operations(self) -> nn.ModuleList:
        """
        """
        lookup_table = layers_lookup()
        layers = nn.ModuleList()
        for layer_name in self.all_layer_names:
            #logger.info('=='*50)
            mod = self.all_layer_dict[layer_name]
            mod_name = mod.__class__.__module__
            class_name = mod.__class__.__name__
            lookup_name = mod_name + '.' + class_name
            lookup_instance = eval(lookup_name)
            if lookup_table[lookup_instance] is not None:
                layers.append(mod)
            else:
                logger.info(f'No Key for {lookup_name}')
            #logger.info('=='*50)
            if isinstance(mod, torch.nn.modules.pooling.AdaptiveAvgPool2d) or isinstance(mod, nn.AvgPool2d):
                synth_flatten = nn.Flatten(start_dim=1)
                setattr(synth_flatten, 'module_name', 'flatten_layer')
                layers.append(synth_flatten)
        return layers
            



    def _create_lrp_model(self) -> nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.operational_layers)
        lookup_table = layers_lookup()
        

        # Run backwards through layers
        # lookup table indices
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer, top_k=self.top_k)
            except Exception as e:
                mismatch_name = layer.__class__.__name__
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{mismatch_name} layer."
                )
                logger.error('**'*50)
                logger.error(e)
                # logger.error(layer.__class__)
                # logger.error(layer)
                # logger.error(f'Undefined: {mismatch_name}')
                logger.error('**'*50)
                self.no_match[mismatch_name] = layer
        return layers
    

    def get_mismatched_layers(self) -> dict:
        return self.no_match
    

    def forward(self, x: torch.tensor,
                synth_relevance: Optional[torch.tensor]=None) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).

        """
        activations = list()
        last_conv_name, last_conv = self.get_last_conv()
        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x)) # why
            #logger.info(len(self.operational_layers))
            #logger.info(self.operational_layers)
            for layer in self.operational_layers:
                try:
                    x = layer.forward(x)
                    if layer.module_name == last_conv.module_name:
                        logger.info(layer.module_name)
                        targ_activations = x
                except Exception as e:
                    logger.error(e)
                activations.append(x)
        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        # Initial relevance scores are the network's output activations
        if synth_relevance is None:
            relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised
            #relevance = torch.rand_like(relevance)
            #shuffle the confidence around
            # permuted_indices = torch.randperm(relevance.size(1)).cuda()
            # relevance = torch.gather(relevance, 1, permuted_indices.unsqueeze(0)).cuda()
        else:
            relevance = synth_relevance
        # inject a fake relevance here
        # Perform relevance propagation
        num_skip = len(self.preceding_layers)
        if synth_relevance is None:
            for i, lrp_layer in enumerate(self.lrp_layers):
                relevance = lrp_layer.forward(activations.pop(0), relevance)
        else:
            for i, lrp_layer in enumerate(self.lrp_layers):
                # look at layers
                if i < num_skip:
                    _ = activations.pop(0)
                elif i == num_skip:
                    _ = activations.pop(0)
                    try:
                        relevance = lrp_layer.forward(activations.pop(0), relevance)
                    except Exception as e:
                        logger.error(e)
                else:
                    try:
                        relevance = lrp_layer.forward(activations.pop(0), 
                                                      relevance)
                    except Exception as e:
                        logger.error(e)
        return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()


    def __call__(self, 
                 x: torch.Tensor,
                 synth_relevance: Optional[torch.Tensor]=None) -> np.ndarray:
        """
        Args:

            x:
            label:

        """
        return self.forward(x=x,
                            synth_relevance=synth_relevance)