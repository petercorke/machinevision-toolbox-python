# animate the display of frames stored within a zip file

from machinevisiontoolbox import FileArchive

# iterate through the zip file, displaying each image in turn, reusing the same window
for image in FileArchive("bridge-l.zip", filter="*.pgm"):
    image.disp(fps=10, reuse=True)

# or, more simply, just display the zip file as an animation
# this version of display has animation controls: [space] pause/resume, [+] faster, [-] slower, [q/x] quit
FileArchive("bridge-l.zip", filter="*.pgm").disp(fps=10)
