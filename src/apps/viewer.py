import argparse
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
    _mod = args.inference[:classname.rfind('.')]
    _class = args.inference[classname.rfind('.')+1:]
    print("Inference:", _mod, _class)
    inference_mod = importlib.import_module("{}.{}".format(prefix, _mod))
    inference_class = getattr(inference_mod, _class)
    return inference_class


def main(args):
    print(args)

    inference_class = load_class(args.inference)
    inference = inference_class()

    input_streams = list(map(create_stream, args.inputs))

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
                inference.handle_keystroke(keystroke)

    finally:
        for stream in input_streams:
            stream.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
