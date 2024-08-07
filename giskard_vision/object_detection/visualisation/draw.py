import cv2
import matplotlib.pyplot as plt


def draw_boxes(image_idx, dataset, model=None):
    """
    Draws predicted and ground truth bounding boxes on the image.

    Args:
        image_idx (int): Index of the image in the dataset.
        dataset (Dataset): The dataset containing images and labels.
        model (Model): The object detection model for making predictions.

    Returns:
        None: This function displays the image with drawn bounding boxes.
    """
    # Retrieve image, labels, and metadata from the dataset
    img, labels, meta = dataset[image_idx]
    pboxes_gt = labels[0]["boxes"]

    # Draw ground truth bounding box (in green)
    cv2.rectangle(
        img[0],
        (int(pboxes_gt[0].item()), int(pboxes_gt[1].item())),
        (int(pboxes_gt[2].item()), int(pboxes_gt[3].item())),
        (0, 255, 0),
        2,
    )

    # Get predictions from the model
    if model:
        predictions = model.predict_image(img[0])
        pboxes = predictions["boxes"]

        # Draw predicted bounding box (in blue)
        cv2.rectangle(
            img[0],
            (int(pboxes[0].item()), int(pboxes[1].item())),
            (int(pboxes[2].item()), int(pboxes[3].item())),
            (255, 0, 0),
            2,
        )

    # Display the image with bounding boxes
    plt.subplots(1, 1, figsize=(20, 20))
    plt.imshow(img[0])
    plt.show()
