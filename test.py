from keras_models import KeypointDetectorModel
from utils.visualization import visualize
from utils.image import denormalize
import cv2 
import numpy as np

# Load image
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image shape: {}".format(image.shape))
visualize(image=image)

# Initialize Keypoint Detector Model
kp_model = KeypointDetectorModel(
    backbone='efficientnetb3', num_classes=29, input_shape=(320, 320),
)

WEIGHTS_PATH = "weights/keypoint_detector.h5"

# Load weights directly from local file path
kp_model.load_weights(WEIGHTS_PATH)

# Get prediction mask
pr_mask = kp_model(image)

# Visualize and save images
visualization_image = denormalize(image.squeeze())
visualization_mask = pr_mask[..., -1].squeeze()
visualization_image_uint8 = (visualization_image * 255).astype(np.uint8)

cv2.imwrite('visualization_image.jpg', cv2.cvtColor(visualization_image_uint8, cv2.COLOR_RGB2BGR))
cv2.imwrite('visualization_mask.jpg', visualization_mask * 255)  # Assuming values are in [0,1]

# Visualize
visualize(
    image=visualization_image,
    pr_mask=visualization_mask,
)
