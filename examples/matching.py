import matplotlib.pyplot as plt
from machinevisiontoolbox import *

# load two different views of the Eiffel tower and compute SIFT features
sf1 = Image.Read("eiffel-1.png").SIFT()
sf2 = Image.Read("eiffel-2.png").SIFT()

# match the features, the result is a list of matching pairs
m = sf1.match(sf2)
print(m[:5])

m[:100].plot("y", width=200, linewidth=0.7)

plt.show(block=True)
