

def get_bbox_from_yolo_format(image_width, image_height, bbox):
    x, y, w, h = bbox
    xmin = int((x - w / 2) * image_width)
    xmax = int((x + w / 2) * image_width)
    ymin = int((y - h / 2) * image_height)
    ymax = int((y + h / 2) * image_height)
    return xmin, ymin, xmax, ymax

def get_yolo_format_from_bbox(image_width, image_height, bbox):
    xmin, ymin, xmax, ymax = bbox
    x = (xmin + xmax) / 2 / image_width
    y = (ymin + ymax) / 2 / image_height
    w = (xmax - xmin) / image_width
    h = (ymax - ymin) / image_height
    return x, y, w, h