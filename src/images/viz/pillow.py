from PIL import Image as PILImage, ImageDraw, ImageFont


def draw_boundingbox(img: PILImage, x: int, y: int, w: int, h: int,
                     colour=(255, 0, 0), width=2, label=None, **kwargs):
    canvas = ImageDraw.Draw(img, "RGBA")
    canvas.rectangle([x, y, x+w, y+h], outline=colour, width=width, **kwargs)

    if label:
        draw_text(img, label, x + w/2, y + h - width, colour=colour)
    return img


def draw_text(img: PILImage, text: str, x: int, y:int, colour=(255, 0, 0)):
    """Draw text, centered at x, bottom-aligned"""
    _font = ImageFont.truetype("FreeMono.ttf", 16)

    canvas = ImageDraw.Draw(img, "RGBA")
    _, _, box_w, box_h = canvas.textbbox((0, 0), text, font=_font)
    canvas.text(((x-box_w/2), (y-box_h)), text, colour, font=_font)
    return img
