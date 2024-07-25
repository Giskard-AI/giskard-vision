import cv2
import matplotlib.pyplot as plt


def draw_boxes(image_idx, dataset, model):
    img, labels, meta = dataset[image_idx]  # norm_image is used for prediction and img for visualisation
    pboxes_gt = labels[0]["boxes"]

    predictions = model.predict_image(img[0])

    pboxes = predictions["boxes"]

    cv2.rectangle(
        img[0],
        (int(pboxes[0].item()), int(pboxes[1].item())),
        (int(pboxes[2].item()), int(pboxes[3].item())),
        (255, 0, 0),
        2,
    )
    cv2.rectangle(
        img[0],
        (int(pboxes_gt[0].item()), int(pboxes_gt[1].item())),
        (int(pboxes_gt[2].item()), int(pboxes_gt[3].item())),
        (0, 255, 0),
        2,
    )

    plt.subplots(1, 1, figsize=(20, 20))
    plt.imshow(img[0])
