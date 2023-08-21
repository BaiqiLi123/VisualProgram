import cv2
import os
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import cv2

import abc
import warnings
from torchvision import transforms
from torch.nn import functional as F
import re

from typing import Union
import pathlib
import json

from lavis.models import load_model_and_preprocess

from transformers import AutoImageProcessor, DPTForDepthEstimation




class XVMLBaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True
    

    def __init__(self, gpu_number):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    # @classmethod
    # @abc.abstractmethod
    # def name(cls) -> str:
    #     """The name of the model has to be given by the subclass"""
    #     pass

    @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPT3Model for an example.
        """
        return [cls.name]