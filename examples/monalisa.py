from machinevisiontoolbox import Image

mona = Image.Read("monalisa.png")
mona.disp(title=f"monalisa from file {mona.name}")

mona = Image.Read("http://petercorke.com/files/images/monalisa.png")
mona.disp(title="monalisa from URL", block=True)
