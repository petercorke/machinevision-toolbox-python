import matplotlib.pyplot as plt
from machinevisiontoolbox import Image
import torchvision as tv

# example from Fig 12.7 of Robotics, Vision & Control, 3rd edition, p. 492

# load the scene image
scene = Image.Read("image3.jpg")


# load a pretrained semantic segmentation model from torchvision
model = tv.models.segmentation.fcn_resnet50(
    weights=tv.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
).eval()

# class labels as per the Pascal VOC dataset, which is what the model was trained on
classes = [
    "Background",
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "TV/Monitor",
]

# push the image through the model to get the segmentation output
in_tensor = scene.tensor(normalize="imagenet")
outputs = model(in_tensor)

fig, (input, output) = plt.subplots(1, 2, figsize=(10, 5))
scene.disp(ax=input)
# display with class labels as colors, and a colorbar showing the mapping from class index to color
Image.Tensor(outputs, logits=True).disp(
    colormap="viridis",
    ncolors=20,
    colorbar=dict(xticks=[range(21), classes]),
    ax=output,
    block=True,
)
