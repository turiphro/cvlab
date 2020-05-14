import argparse
import cv2
import importlib
from typing import Sequence

from streams import create_stream
from images.image_type import ImageType


argparse.ArgumentDefaultsHelpFormatter
WINDOW_NAME = 'cvlab::viewer'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '--input', '-i', nargs='+', default=["0"],
                        help='Input stream or streams (camera id:int, image path:str)')
    parser.add_argument('--inference', '-f', type=load_class, default=load_class("filters.Nothing"),
                        help="Inference class to use")

    # Remaining arguments are collected for parse_additional_args
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def parse_additional_args(input: Sequence[str], arg_defs: dict):
    """Parse class-specific arguments"""
    parser = argparse.ArgumentParser()

    shorthands = {'i', 'f'}
    for _key, _type in arg_defs.items():
        args = ['--{}'.format(_key)]
        if _key[0] not in shorthands:
            args.append('-{}'.format(_key[0]))
            shorthands.add(_key[0])
        parser.add_argument(*args, type=_type)

    args = parser.parse_args(input)
    return args


def load_class(classname: str, prefix="inference"):
    _mod, _class = classname.rsplit('.', 1)
    inference_mod = importlib.import_module("{}.{}".format(prefix, _mod))
    inference_class = getattr(inference_mod, _class)
    return inference_class


def main(args: argparse.Namespace, class_args: argparse.Namespace):
    print(args)
    print(class_args)

    inference = args.inference(**vars(class_args))

    input_streams = list(map(create_stream, args.inputs))

    print("Inference:", args.inference)
    print("| Shortcuts available:")
    print("|  q  quit")
    for shortcut, descr in inference.KEYSTROKES.items():
        print("|  {}  {}".format(shortcut, descr))

    try:
        while True:
            input_images = list(stream.get(ImageType.OPENCV) for stream in input_streams)
            output_images = inference.process(input_images)

            for name, image in output_images.items():
                if image is not None:
                    frame_image = image.get(ImageType.OPENCV)
                    cv2.imshow(WINDOW_NAME + "::{}".format(name), frame_image)

            keystroke = chr(cv2.waitKey(1) & 0xFF)
            if keystroke == 'q':
                break
            elif keystroke in inference.KEYSTROKES:
                inference.handle_command(keystroke)

    finally:
        for stream in input_streams:
            stream.stop()


if __name__ == '__main__':
    args, unknown_args = parse_args()
    class_args = parse_additional_args(unknown_args, args.inference.ARGUMENTS)
    main(args, class_args)
