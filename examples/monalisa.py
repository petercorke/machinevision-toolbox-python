from machinevisiontoolbox import Image

mona = Image.Read("monalisa.png")
mona.disp(title=f"monalisa from file {mona.name}")

mona = Image.Read("https://raw.githubusercontent.com/petercorke/machinevision-toolbox-python/main/packages/mvtb-data/mvtbdata/images/monalisa.png")
mona.disp(title="monalisa from URL", block=True)
