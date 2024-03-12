import cv2
import numpy as np

def draw_box(
        image: np.ndarray,
        box: tuple,
        label: str = '',
        color: tuple = (128, 128, 128),
        txt_color: tuple = (255, 255, 255),
):
    """Draw box and label on image.

    Args:
        image (numpy.ndarray): The image to draw on.
        box (tuple): The box to draw. (format [x1, y1, x2, y2])
        label (str, optional): The label of the box. Defaults to ''.
        color (tuple, optional): The box color. Defaults to (128, 128, 128).
        txt_color (tuple, optional): The text color. Defaults to (255, 255, 255).
    """
    lt = max(round(sum(image.shape) / 2 * 0.003), 2)  # line thickness
    p1 = int(box[0]), int(box[1])
    p2 = int(box[2]), int(box[3])
    c = tuple(color)
    cv2.rectangle(
        image, p1, p2, c,
        thickness=lt,
        lineType=cv2.LINE_AA
    )
    if label:
        ft = max(lt - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, lt / 3, ft)[0]  # text width, height

        outside = p1[1] - h >= 3
        x, y = p1[0], p1[1] - 2 if outside else p1[1] + h + 2
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(  # label background
            image, p1, p2, c,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image, label, (x, y), 0, lt / 3, txt_color,
            thickness=ft,
            lineType=cv2.LINE_AA,
        )


def draw_bboxes(
        image: np.ndarray,
        boxes: np.ndarray,
        track_ids: np.ndarray = None,
        labels: list = [],
        colors: list = [],
        score=True,
        conf=None,
):
    """Draw boxes on image.

    Args:
        image (numpy.ndarray): The image to draw on.
        boxes (numpy.ndarray): The boxes to draw. (format [x1, y1, x2, y2, conf, cls])
        track_ids (numpy.ndarray, optional): The track ids of the boxes. Defaults to None.
        labels (list, optional): The labels of the boxes. Defaults to [].
        colors (list, optional): The colors of the boxes. Defaults to [].
        score (bool, optional): Whether to draw the score of the boxes. Defaults to True.
        conf ([type], optional): The confidence threshold. Defaults to None.
    """
    # Define colors.
    if colors == []:
        np.random.seed(42)
        colors = np.random.randint(
            0, 255,
            size=(len(labels), 3),
            dtype="uint8"
        )
        colors = [tuple(c) for c in colors.tolist()]
    color = (128, 128, 128)

    # Draw boxes.
    for i, box in enumerate(boxes):
        text = []

        if conf and box[-2] < conf:
            continue

        if labels != []:
            text.append(labels[int(box[-1])])
            color = colors[int(box[-1])]

        if score:
            score = float(box[-2]) * 100
            text.append(f'{score:.1f}%')

        if track_ids is not None:
            text.append(f'#{int(track_ids[i])}')

        text = ' '.join(text)

        draw_box(image, box, text, color)