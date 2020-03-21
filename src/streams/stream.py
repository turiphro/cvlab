from images.image import Image


class Stream:
    """Interface for streams (sources) of Images"""
    identifier = None

    def __init__(self, identifier):
        self.identifier = identifier

    def get(self, latest=False) -> Image:
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()
