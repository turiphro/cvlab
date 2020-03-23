from typing import Dict, Any
import mxnet as mx
import gluoncv as gcv
import numpy as np
import cv2

from images.image import Image
from .loader import ModelLoader, ModelType


FRIENDLY_NAMES = {
    # See gluoncv.model_zoo.model_zoo.py:_models for the full pretrained gluoncv list

    # CLASSIFICATION
    # https://gluon-cv.mxnet.io/model_zoo/classification.html
    "resnet": ("resnet152_v2", ModelType.CLASSIFICATION),
    "resnext": ("se_resnext50_32x4d", ModelType.CLASSIFICATION),
    "mobilenet": ("mobilenet0.75", ModelType.CLASSIFICATION),

    # DETECTION
    # https://gluon-cv.mxnet.io/model_zoo/detection.html
    "yolo": ("yolo3_darknet53_coco", ModelType.DETECTION),
    "ssd": ("ssd_512_mobilenet1.0_coco", ModelType.DETECTION),

    # SEGMENTATION
    # https://gluon-cv.mxnet.io/model_zoo/segmentation.html
    "fcn": ("fcn_resnet101_voc", ModelType.SEGMENTATION),
    "deeplab": ("psp_resnet101_coco", ModelType.SEGMENTATION),
    #
}


class MxnetLoader(ModelLoader):
    def __init__(self):
        self.model = None
        self.model_type = None

    def load(self, model_name: str, model_type: ModelType = None) -> None:
        _model_name, _model_type = FRIENDLY_NAMES.get(model_name) or (model_name, model_type)
        self.model = gcv.model_zoo.get_model(_model_name, pretrained=True)
        self.model.hybridize()
        self.model_type = _model_type

    def process(self, image: Image, short=512, max_size=640,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("[mxnet] No model has been loaded. Run load() first.")

        # preprocess
        np_img = image.asnumpy()
        mx_img = mx.nd.array(np_img).astype('uint8')
        mx_img = gcv.data.transforms.image.resize_short_within(mx_img, short=short, max_size=max_size, mult_base=32)
        ts_img = mx.nd.image.to_tensor(mx_img)
        ts_img = mx.nd.image.normalize(ts_img, mean=mean, std=std).expand_dims(0)

        # run model
        if hasattr(self.model, "predict"):
            # Some models have this method (i.e., segmentation models), some don't
            outputs = self.model.predict(ts_img)
        else:
            outputs = self.model(ts_img)

        if self.model_type is ModelType.CLASSIFICATION:
            return {
                "class_ids": outputs
            }
        elif self.model_type is ModelType.DETECTION:
            return {
                "class_ids": outputs[0],
                "scores": outputs[1],
                "bounding_boxes": outputs[2]
            }
        elif self.model_type is ModelType.SEGMENTATION:
            return {
                "mask": outputs
            }
        else:
            return {
                "outputs": outputs
            }

    def visualise(self, image: Image, metadata: Dict[str, Any], blend=0.5, show_labels=True):
        np_img = image.asnumpy()
        mx_img = mx.nd.array(np_img).astype('uint8')

        img = None
        if all(key in metadata for key in ["bounding_boxes", "scores", "class_ids"]):
            # object detection results
            bounding_boxes = metadata["bounding_boxes"]
            scores = metadata["scores"]
            class_ids = metadata["class_ids"]

            img = gcv.utils.viz.cv_plot_bbox(
                mx_img, bounding_boxes[0], scores[0], class_ids[0], class_names=self.model.classes)

        elif "class_ids" in metadata:
            # classification results
            class_ids = metadata["class_ids"]
            scores = mx.nd.softmax(class_ids)[0].asnumpy()

            top_ids = mx.nd.topk(class_ids, k=3)[0].astype("int").asnumpy()
            captions = ["{}: {:.3}".format(self.model.classes[id], scores[id])
                        for id in top_ids]
            img = draw_captions(np_img, captions) if show_labels else np_img

        elif "mask" in metadata:
            # segmentation results
            mask = metadata["mask"]
            img_resized = gcv.data.transforms.image.imresize(mx_img, mask.shape[3], mask.shape[2]).asnumpy()

            predict = mx.nd.argmax(mask, 1)[0].astype("int").asnumpy()
            mask_colour = gcv.utils.viz.get_color_pallete(predict, "pascal_voc")
            mask_colour = np.array(mask_colour.convert("RGB")) # PIL with palette -> numpy RGB
            img = cv2.addWeighted(img_resized, 1 - blend, mask_colour, blend, 0)

            if show_labels:
                classfreqs = list(zip(*np.unique(predict, return_counts=True)))
                classfreqs = sorted(classfreqs, key=lambda cf: cf[1], reverse=True)
                # NOTE: a bug in gluoncv currently returns a property of a class, instead of an instance of the class,
                #       for all Segmentation models; we need to obtain the property value (passing a bogus self)
                model_classes = self.model.classes.fget(0)
                captions = ["{} ({}px)".format(model_classes[id], freq)
                            for (id, freq) in classfreqs[1:5]]
                img = draw_captions(img, captions)

        else:
            raise ValueError(
                "Don't know how to visualise metadata with keys " + ",".join(metadata.keys()))

        img_result = Image(img, opencv=False)
        return img_result


def draw_captions(img: np.ndarray, captions, colour=(255, 255, 255), thickness=3, scale=1,
                  offset=(20, 20), font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw one or more strings at the bottom of the image"""
    for i, caption in enumerate(reversed(captions)):
        location = (offset[0], img.shape[0] - 40*i - offset[1])
        cv2.putText(img, caption, location,
                    fontFace=font, fontScale=scale, color=colour, thickness=thickness)

    return img
