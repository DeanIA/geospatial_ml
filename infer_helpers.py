import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

def load_yolo_obb_as_xyxy(label_path, image_path):
    """
    Read image size + YOLO-OBB labels → axis-aligned pixel boxes.

    Args:
      label_path:  path to YOLO-OBB .txt (cls + coords normalized 0–1)
      image_path:  path to the image file

    Returns:
      (N,4) array of [x_min, y_min, x_max, y_max] in pixel coords.
    """
    # 1) load image and get its width/height
    img = Image.open(image_path)
    img_width, img_height = img.size

    boxes = []
    with open(label_path, 'r') as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            vals = list(map(float, parts[1:]))

            if len(vals) == 8:
                # polygon format: scale each corner
                xs = np.array(vals[0::2]) * img_width
                ys = np.array(vals[1::2]) * img_height
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

            elif len(vals) == 5:
                # xywha: scale center + dims
                x_c, y_c, w, h, theta = vals
                x_c *= img_width
                y_c *= img_height
                w   *= img_width
                h   *= img_height

                dx, dy = w/2, h/2
                rel = np.array([[-dx,-dy],[-dx,dy],[dx,dy],[dx,-dy]])
                R   = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
                corners = rel @ R.T + np.array([x_c, y_c])
                x_min, y_min = corners[:,0].min(), corners[:,1].min()
                x_max, y_max = corners[:,0].max(), corners[:,1].max()

            else:
                raise ValueError(f"Line {i}: expected 5 or 8 coords, got {len(vals)}")

            boxes.append([x_min, y_min, x_max, y_max])

    return np.array(boxes, dtype=float)

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    import matplotlib.pyplot as plt
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()