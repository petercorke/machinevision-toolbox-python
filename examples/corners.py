from machinevisiontoolbox import Image
import matplotlib.pyplot as plt

# load an image, Fig 14.1a in RVC3e
im = Image.Read("eiffel-1.png")
print("Eiffel tower:", im)

## compute SIFT features
features = im.SIFT()

# print the summary
print(features)

# print the first 10 features
for feature in features[:10]:
    print(feature)

# overlay the 50 largest features on the image, sorted by decreasing strength
im.disp(title="Eiffel tower + SIFT features", block=False)
features.sort().filter(minscale=10)[:50].plot(filled=True, color="red", alpha=0.5)

## compute ORB features, these have no scale
features = im.ORB()
# print the summary
print(features)

# print the first 10 features
for feature in features[:10]:
    print(feature)

# overlay the 50 largest features on the image, sorted by decreasing strength
im.disp(title="Eiffel tower + ORB features", block=False)
features.sort()[:50].plot(color="yellow", alpha=0.3)

plt.show(block=True)
