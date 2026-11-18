import numpy as np

from machinevisiontoolbox import BagOfWords, FileCollection, Image
from machinevisiontoolbox.ImagePointFeatures import BaseFeature2D


def accepted(
    bag: BagOfWords, image: Image, features: BaseFeature2D, files: FileCollection
) -> None:
    BagOfWords(features)
    BagOfWords([image])
    BagOfWords(iter([image]))
    BagOfWords(None)
    bag.similarity(image)
    bag.similarity([image])
    bag.similarity(files)
    bag.similarity(np.zeros((2, 1)))
    bag.retrieve(image)
    bag.retrieve(files)
    BagOfWords._word_freq_vector([0, 1], 2)
