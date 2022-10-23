import boto3
import time
from botocore.client import Config
from typing import Dict, Any, Tuple
from random import randint

from images.image import Image
from images.viz.pillow import draw_boundingbox, draw_point
from .provider import CloudProvider, InferenceType


class AWSInference(CloudProvider):
    """
    AWS Rekognition on image frames
    
    NOTE: this class runs Rekognition on a frame-by-frame basis.
    AWS also supports running on video streams for some of the APIs,
    requiring a Kinesis Video Stream integration (not implemented):
    - https://docs.aws.amazon.com/rekognition/latest/dg/streaming-video.html
    - https://docs.aws.amazon.com/kinesisvideostreams/latest/dg/what-is-kinesis-video.html
    """
    KEYSTROKES = {'r': "Reset tracking"}

    def __init__(self):
        self.COLOURS = {}

    def load(self, inference_type: InferenceType, debug: int = True) -> None:
        config = Config(connect_timeout=1, read_timeout=3)
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
        _start = time.perf_counter()

        if self.inference_type == InferenceType.DETECTION:
            rekognition_response = self.rekognition_client.detect_labels(
                Image=image2rekimage(image),
                MinConfidence=50,
            )
            return {
                "detections": rekognition_response["Labels"],
                "latency": time.perf_counter() - _start,
            }

        elif self.inference_type == InferenceType.FACE_DETECTION:
            rekognition_response = self.rekognition_client.detect_faces(
                Image=image2rekimage(image),
                Attributes=["ALL"],
            )
            return {
                "faces": rekognition_response["FaceDetails"],
                "latency": time.perf_counter() - _start,
            }

        elif self.inference_type == InferenceType.TEXT_EXTRACT:
            rekognition_response = self.rekognition_client.detect_text(
                Image=image2rekimage(image),
                Filters={"WordFilter": {"MinConfidence": 50}},
            )
            return {
                "text": rekognition_response["TextDetections"],
                "latency": time.perf_counter() - _start,
            }

        else:
            raise NotImplementedError(
                f"Unknown inference type inside {self.__class__.__name__}: {self.inference_type}"
            )

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
                print(f"!! Error parsing detections, skipping: {ex}")

        if "faces" in metadata:
            try:
                parsed = parse_rek_faces(metadata["faces"], img.size)

                for i, face in enumerate(parsed):
                    _key = f"face{i}"
                    if _key not in self.COLOURS:
                        # assigning colours assuming the detection order stays (relatively) stable
                        self.COLOURS[_key] = (randint(0, 255), randint(0, 255), randint(0, 255))

                    _colour = self.COLOURS[_key]
                    draw_boundingbox(img, *face["boundingbox"], colour=_colour, label=_key)
                    for label, (x, y) in face["landmarks"].items():
                        draw_point(img, x, y, label=None, colour=_colour)

            except ValueError as ex:
                print(f"!! Error parsing detections, skipping: {ex}")

        if "text" in metadata:
            try:
                parsed = parse_rek_text(metadata["text"], img.size)

                for i, text in enumerate(parsed):
                    _colour = (
                        int(255 * (100 - text["confidence"]) / 100),
                        int(255 * text["confidence"] / 100),
                        0
                    )
                    draw_boundingbox(img, *text["boundingbox"], colour=_colour, label=text["label"])

            except ValueError as ex:
                print(f"!! Error parsing detections, skipping: {ex}")

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


def parse_rek_faces(raw: dict, img_size: Tuple[int, int]) -> dict:
    """
    Example input:
        [
            {'Confidence': 98.98920440673828,
             'BoundingBox':
                {'Height': 0.7440704107284546,
                 'Left': 0.3230002522468567,
                 'Top': 0.24780337512493134,
                 'Width': 0.612679660320282},
                'Confidence': 98.98920440673828},
             '<Property>': {'Value': true, 'Confidence': 98.2},
             'Pose': {'Roll': 11.1 'Yaw': 31.6, 'Pitch': 11.2}
             'Landmarks': [
                {'Type': "eyeLeft", 'X': 0.52, 'Y': 0.24}
             ]
            },
        ]

    Example output:
        [
            {'confidence': 98.9,
             'boundingbox': (32, 24, 61, 74),  // (x, y, w, h)
             'landmarks': {'eyeLeft': (520, 240), ..},
            },
        ]
    """
    # NOTE: ignoring the Emotions and the various properties for now
    MANDATORY_KEYS = ["Confidence", "BoundingBox", "Landmarks"]

    output = []
    for item in raw:
        if any(label not in item for label in MANDATORY_KEYS):
            raise ValueError(f"One or more keys from {MANDATORY_KEYS} missing in {item}")
        boundingbox = (
            int(item["BoundingBox"]["Left"] * img_size[0]),
            int(item["BoundingBox"]["Top"] * img_size[1]),
            int(item["BoundingBox"]["Width"] * img_size[0]),
            int(item["BoundingBox"]["Height"] * img_size[1])
        )
        landmarks = {
            lm["Type"]: (lm["X"] * img_size[0], lm["Y"] * img_size[1])
            for lm in item["Landmarks"]
        }
        output.append({
            "confidence": item["Confidence"],
            "boundingbox": boundingbox,
            "landmarks": landmarks,
        })
    return output


def parse_rek_text(raw: dict, img_size: Tuple[int, int]) -> dict:
    """
    Example input:
        [
            {
                "DetectedText": "some word",
                "Confidence": 8.662981986999512,
                "Type": "WORD",  OR "LINE"
                "Id": 0,
                "ParentId": 0,
                "Geometry": {
                    "BoundingBox": {
                        "Width": 0.30775646371976645,
                        "Height": 0.044534412955465584,
                        "Left": 0.36363636363636365,
                        "Top": 0.05668016194331984
                    },
                    "Polygon": [...]
                },
            }
        ]

    Example output: (only includes WORD entries)
        [
            {'label': "some word",
             'confidence': 98.9,
             'boundingbox': (32, 24, 61, 74),  // (x, y, w, h)
            },
        ]
    """
    # NOTE: will filter out LINEs; ignoring the polygon for now
    MANDATORY_KEYS = ["Confidence", "Geometry", "DetectedText"]

    output = []
    for item in raw:
        if any(label not in item for label in MANDATORY_KEYS):
            raise ValueError(f"One or more keys from {MANDATORY_KEYS} missing in {item}")
        boundingbox = (
            int(item["Geometry"]["BoundingBox"]["Left"] * img_size[0]),
            int(item["Geometry"]["BoundingBox"]["Top"] * img_size[1]),
            int(item["Geometry"]["BoundingBox"]["Width"] * img_size[0]),
            int(item["Geometry"]["BoundingBox"]["Height"] * img_size[1])
        )
        if item["Type"] == "WORD":
            output.append({
                "confidence": item["Confidence"],
                "label": item["DetectedText"],
                "boundingbox": boundingbox,
            })
    return output
