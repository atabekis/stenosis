# visualize.py

# Python imports
import cv2
import imageio
import matplotlib.pyplot as plt

from config import CADICA_DATASET_DIR, DANILOV_DATASET_DIR
# Local imports
from methods.reader import XCAImage, Reader


def overlay_bbox(image: XCAImage, bboxes: list, color=(0, 255, 0), thickness=1):
    """
    Overlay bounding boxes on the image.
    """
    img = image.copy()

    if bboxes and isinstance(bboxes[0], (list, tuple)): # Danilov format
        for box in bboxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        return img

    else: # CADICA format
        xmin, ymin, xmax, ymax = bboxes
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        return img



def show_xca_image(xca_image: XCAImage, overlay: bool = True, figsize=(6, 6)):
    """
    Display an XCAImage instance
    """
    image = xca_image.get_image()

    image = overlay_bbox(image, xca_image.bbox) if overlay else image

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.title(f"Patient={xca_image.patient_id}, Video={xca_image.video_id}, Frame={xca_image.frame_nr}")
    plt.axis('off')
    plt.show()


def image_to_gif(xca_images, output_path="output.gif", overlay=True, fps=1):
    """Construct GIF from a list of XCAImage instances."""
    frames = []
    for xca_image in xca_images:
        image = xca_image.get_image()
        image = overlay_bbox(image, xca_image.bbox) if overlay else image

        text = f"Patient: {xca_image.patient_id}, Video: {xca_image.video_id}, Frame: {xca_image.frame_nr}"
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 0.5
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = 10
        text_y = image.shape[0] - 10
        cv2.rectangle(image, (text_x, text_y - text_size[1] - baseline),
                      (text_x + text_size[0], text_y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image_rgb)

    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


if __name__ == "__main__":
    r = Reader(dataset_dir=CADICA_DATASET_DIR)
    ins = r.get(lesion=True)
    show_xca_image(ins)