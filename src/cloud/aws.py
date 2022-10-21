import boto3
import time
from botocore.client import Config
from typing import Dict, Any, Tuple
from random import randint

from images.image import Image
from images.viz.pillow import draw_boundingbox, draw_text
from .provider import CloudProvider, InferenceType


class AWSInference(CloudProvider):
    KEYSTROKES = {'r': "Reset tracking"}

    def __init__(self):
        self.COLOURS = {}

    def load(self, inference_type: InferenceType, debug: int = True) -> None:
        config = Config(connect_timeout=1, read_timeout=2)
        self.rekognition_client = boto3.client("rekognition", config=config)
        self.inference_type = inference_type
        self.debug = debug

        print()
        print("!! WARNING: this inference type uses AWS Rekognition,")
        print("!!          which can add up to significant cost when running")
        print("!!          for longer periods of time ($250/day for 3 TPS)")
        print()
        input("Hit ENTER to confirm >>> ")

    def process(self, image: Image) -> Dict[str, Any]:
        if self.inference_type == InferenceType.DETECTION:
            _start = time.perf_counter()
            rekognition_response = self.rekognition_client.detect_labels(
                Image=image2rekimage(image),
                MinConfidence=50,
            )
            return {
                "detections": rekognition_response["Labels"],
                "latency": time.perf_counter() - _start,
            }

        else:
            raise NotImplementedError(f"Unknown inference type: {self.inference_type}")

    def visualise(self, image: Image, metadata: Dict[str, Any]) -> Image:
        img = image.aspil()

        if "detections" in metadata:
            try:
                parsed = parse_rek_detect(metadata["detections"], img.size)

                for item in parsed:
                    if item["label"] not in self.COLOURS:
                        self.COLOURS[item["label"]] = (randint(0, 255), randint(0, 255), randint(0, 255))
                    for instance in item["instances"]:
                        _colour = self.COLOURS[item["label"]]
                        draw_boundingbox(img, *instance, colour=_colour, label=item["label"])
                if self.debug:
                    print("----classes-found-----")
                    for item in sorted(parsed, key=lambda x: x["label"]):
                        print(f' {item["label"]}: {item["confidence"]}')
            except ValueError as ex:
                print("!! Error parsing detections, skipping: ex")

        if self.debug and "latency" in metadata:
            print(f'=> latency: {metadata["latency"]}')

        img_result = Image(img)
        return img_result

    def handle_command(self, key):
        if key == 'r':
            print("Resetting colours")
            self.COLOURS = {}



def image2rekimage(image: Image) -> dict:
    image_bytes = image.asbytes()
    return {"Bytes": image_bytes}


def parse_rek_detect(raw: dict, img_size: Tuple[int, int]) -> dict:
    """
    Example input:
        [
            {'Confidence': 98.98920440673828,
             'Instances': [{'BoundingBox':
                {'Height': 0.7440704107284546,
                 'Left': 0.3230002522468567,
                 'Top': 0.24780337512493134,
                 'Width': 0.612679660320282},
                'Confidence': 98.98920440673828}],
             'Name': 'Person',
             'Parents': []
            },
            {'Confidence': 50.30917739868164,
             'Instances': [],
             'Name': 'Cottage',
             'Parents': [{'Name': 'House'}, {'Name': 'Housing'}, {'Name': 'Building'}]
            },
        ]

    Example output:
        [
            {'label': 'Person',
             'confidence': 98.9,
             'instances': [(32, 24, 61, 74)]  // (x, y, w, h)
            },
            {'label': 'Building/Housing/House',
             'confidence': 50.3,
             'instances': []
            },
        ]
    """
    MANDATORY_KEYS = ["Confidence", "Instances", "Name", "Parents"]

    output = []
    for item in raw:
        if any(label not in item for label in MANDATORY_KEYS):
            raise ValueError(f"One or more keys from {MANDATORY_KEYS} missing in {item}")
        label_parts = list(reversed(item["Parents"])) + [item]
        label = "/".join(p["Name"] for p in label_parts)
        instances = [
            (
                int(inst["BoundingBox"]["Left"] * img_size[0]),
                int(inst["BoundingBox"]["Top"] * img_size[1]),
                int(inst["BoundingBox"]["Width"] * img_size[0]),
                int(inst["BoundingBox"]["Height"] * img_size[1])
            )
            for inst in item["Instances"]
            if "BoundingBox" in inst
        ]
        output.append({
            "label": label,
            "confidence": item["Confidence"],
            "instances": instances,
        })
    return output
