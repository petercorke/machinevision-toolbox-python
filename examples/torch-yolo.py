import matplotlib.pyplot as plt
from machinevisiontoolbox import Image, plot_labelbox

# example from Fig 12.8 of Robotics, Vision & Control, 3rd edition, p. 508, but using
# the latest YOLO26 nano model from the ultralytics library, which is very fast and
# NMS-free. This example demonstrates how to use a pretrained object detection model to
# detect objects in an image and display the results with bounding boxes and class
# labels.

# we need to install the ultralytics library to run this example, which provides a PyTorch implementation of the YOLO object detection model
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics YOLO library not found. Please install it with 'pip install ultralytics' to run this example."
    )


# Load the latest YOLO26 nano model (very fast, NMS-free)
model = YOLO("yolo26n.pt")

# load a street scene image and display it
img = Image.Read("image3.jpg")
img.disp()

# Run inference on the Numpy array of the image
results = model(img.array)

# Returns a list of results (one per image, but we only have one image here)
#  we need to get the data back onto the CPU and convert to Numpy arrays for use with the plotting functions in machinevisiontoolbox
boxes = results[0].boxes.xyxy.cpu().numpy()  # [xmin, ymin, xmax, ymax]
classes = results[0].boxes.cls.cpu().numpy()

for box, class_id in zip(boxes, classes):
    plot_labelbox(f"{model.names[class_id]}", lbrt=box, color="red")

plt.show(block=True)
