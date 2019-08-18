import argparse
import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import cv2
import importlib

from streams import create_stream
from images.image_type import ImageType


WINDOW_NAME = 'cvlab::viewer'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '--input', '-i', nargs='+',
                        help='Input stream or streams (camera id:int, image path:str)')
    parser.add_argument('--inference', '-f', default='filters.Nothing',
                        help="Inference class to use")

    args = parser.parse_args()
    return args


def load_class(classname, prefix="inference"):
    _mod = args.inference[:args.inference.rfind('.')]
    _class = args.inference[args.inference.rfind('.')+1:]
    print("Inference:", _mod, _class)
    inference_mod = importlib.import_module("{}.{}".format(prefix, _mod))
    inference_class = getattr(inference_mod, _class)
    return inference_class


def main(args):
    print(args)

    inference_class = load_class(args.inference)
    inference = inference_class()

    input_streams = list(map(create_stream, args.inputs))

    try:
        while True:
            input_images = list(stream.get(ImageType.OPENCV) for stream in input_streams)
            output_images = inference.process(input_images)

            for name, image in output_images.items():
                frame_image = image.get(ImageType.OPENCV)
                if frame_image is not False:
                    cv2.imshow(WINDOW_NAME + "::{}".format(name), frame_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for stream in input_streams:
            stream.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
