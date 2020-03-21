from images.image import Image

from typing import Sequence, Dict


class Inference:
    """Interface for Inference classes"""
    KEYSTROKES = {}  # Set to {character: description} to be handled by handle_keystroke

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        """
        The process method takes an ordered list of images (each Image or None),
        and produces a dict with named output images.
        """
        raise NotImplementedError()

    def handle_command(self, key):
        """
        Handle single-character commands (e.g., keyboard strokes for the interactive viewer)
        """
        pass  # optional to implement
