import cv2 as cv
from ansitable import ANSITable, Column
from machinevisiontoolbox.base import color_bgr
import matplotlib.pyplot as plt
import numpy as np
import spatialmath.base as smb
from collections.abc import Iterable


def draw_box(image,
    l=None,
    r=None,
    t=None,
    b=None,
    w=None,
    h=None,

    lb=None,
    lt=None,
    rb=None,
    rt=None,
    wh=None,
    centre=None,

    ax=None,
    bbox=None,
    ltrb=None,
    color=None, thickness=1):
    """
    Draw a box in an image

    :param image: image to draw into, greyscale or color
    :type image: ndarray(H,W), ndarray(H,W,P)

    :param l: left side coordinate
    :type l: int, optional
    :param r: right side coordinate
    :type r: int, optional
    :param t: top side coordinate
    :type t: int, optional
    :param b: bottom side coordinate
    :type b: int, optional

    :param bbox: bounding box [xmin, xmax, ymin, ymax]
    :type bbox: array_like(4), optional
    :param ltrb: bounding box [xmin, ymin, xmax, ymax] 
    :type ltrb: array_like(4), optional

    :param lb: left-bottom corner [x,y]
    :type lb: array_like(2), optional
    :param lt: left-top corner [x,y]
    :type lt: array_like(2), optional
    :param rb: right-bottom corner (x,y)
    :type rb: array_like(2), optional
    :param rt: right-top corner (x,y)
    :type rt: array_like(2), optional
    :param wh: width and height
    :type wh: array_like(2), optional
    :param centre: box centre (x,y)
    :type centre: array_like(2), optional

    :param color: color of line
    :type color: scalar or array_like
    :param thickness: line thickness, -1 to fill, defaults to 1
    :type thickness: int, optional
    :param ax: axes to draw into
    :type: Matplotlib axes

    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    Draws a box into the specified image using OpenCV.  The input ``image``
    is modified.

    The box can be specified in many ways, any combination of inputs is allowed
    so long as the box is fully specified:

    - bounding box [xmin, xmax; ymin, ymax]
    - left-top-right-bottom [xmin, ymin, xmax, ymax]
    - left side
    - right side
    - top side
    - bottom side
    - centre and width+height
    - left-bottom and right-top corners
    - left-bottom corner and width+height
    - right-top corner and width+height
    - left-top corner and width+height

    where left-bottom is (xmin, ymin), left-top is (xmax, ymax)

    Example::

        >>> from machinevisiontoolbox import draw_box, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_box(img, ltrb=[100, 300, 700, 500], thickness=2, color=200)
        >>> draw_box(img, ltrb=[100, 300, 700, 500], thickness=-1, color=50)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_box, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_box(img, ltrb=[100, 300, 700, 500], thickness=2, color=200)
        draw_box(img, ltrb=[100, 300, 700, 500], thickness=-1, color=50)
        idisp(img)

    .. note::
        - For images y increases downwards so :math:`y_{top} < y_{bottom}`
        - if ``image`` is a 3-plane image then ``color`` should be a 3-vector
          or colorname string and the corresponding elements are used in 
          each plane. 

    :seealso: :func:`~smtb.base.graphics.plot_box`  `opencv.rectangle <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9>`_
    """

    # if not isinstance(color, (int, float)) and len(image.shape) == 2:
    #     raise TypeError("can't draw color into a greyscale image")

    if bbox is not None:
        if isinstance(bbox, np.ndarray) and bbox.ndims > 1:
            # case of [l r; t b]
            bbox = bbox.ravel()
        l, r, t, b = bbox
    elif ltrb is not None:
        l, t, r, b = ltrb
    else:
        if lt is not None:
            l, t = lt
        if rt is not None:
            r, t = rt
        if lb is not None:
            l, b = lb
        if rb is not None:
            r, b = rb
        if wh is not None:
            if isinstance(wh, Iterable):
                w, h = wh
            else:
                w = wh
                h = wh
        if centre is not None:
            cx, cy = centre

        if l is None:
            try:
                l = r - w
            except:
                pass
        if l is None:
            try:
                l = cx - w / 2
            except:
                pass

        if r is None:
            try:
                r = l + w
            except:
                pass
        if r is None:
            try:
                r = cx + w / 2
            except:
                pass
        
        if t is None:
            try:
                t = b + h
            except:
                pass
        if t is None:
            try:
                t = cy + h / 2
            except:
                pass

        if b is None:
            try:
                b = t - h
            except:
                pass
        if b is None:
            try:
                b = cy - h / 2
            except:
                pass

    if l >= r:
        raise ValueError("left must be less than right")
    if b >= t:
        #raise ValueError("bottom must be less than top")
        b, t = t, b
        
    # TODO need to do this?
    bl = tuple([int(x) for x in (l, b)])
    tr = tuple([int(x) for x in (r, t)])

    if isinstance(color, str):
        color = color_bgr(color)

    cv.rectangle(image, bl, tr, color, thickness)

    return bl, tr


def plot_labelbox(text, textcolor=None, labelcolor=None, **boxargs):
    """
    Plot a labelled box using matplotlib

    :param text: text label
    :type text: str
    :param textcolor: text color, defaults to None
    :type textcolor: str, array_like(3), optional
    :param labelcolor: label background color
    :type labelcolor: str, array_like(3), optional
    :param boxargs: arguments passed to :func:`plot_box`

    Plot a box with a label above it. The position of the box is specified using
    the same arguments as for ``plot_box``. The label font is specified using
    the same arguments as for ``plot_text``.  If ``labelcolor`` is specified it
    is used as the background color for the text label, otherwise the box color
    is used.

    Example::

        >>> from machinevisiontoolbox import plot_labelbox
        >>> plot_labelbox('labelled box', bbox=[100, 150, 300, 350], color='r')

    .. plot::

        from machinevisiontoolbox import plot_labelbox
        from spatialmath.base import plotvol2
        plotvol2([0, 1000])
        plot_labelbox('labelled box', bbox=[100, 150, 300, 350], color='r'


    .. note:: The label is drawn at the top of the box assuming that axes
        are drawn with the y-axis downward (image convention).

    :seealso: :func:`~spatialmath.base.plot_box`, :func:`~spatialmath.base.plot_text`
    """

    rect = smb.plot_box(**boxargs)

    bbox = rect.get_bbox()

    if labelcolor is None:
        labelcolor = boxargs.get('color')
    smb.plot_text((bbox.xmin, bbox.ymin), text, color=textcolor, verticalalignment='bottom', 
        bbox=dict(facecolor=labelcolor, linewidth=0, edgecolor=None))

_fontdict = {
    'simplex': cv.FONT_HERSHEY_SIMPLEX, 
    'plain': cv.FONT_HERSHEY_PLAIN, 
    'duplex': cv.FONT_HERSHEY_DUPLEX, 
    'complex': cv.FONT_HERSHEY_COMPLEX, 
    'triplex': cv.FONT_HERSHEY_TRIPLEX, 
    'complex-small': cv.FONT_HERSHEY_COMPLEX_SMALL, 
    'script-simplex': cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 
    'script-complex': cv.FONT_HERSHEY_SCRIPT_COMPLEX, 
    'italic': cv.FONT_ITALIC,         
}

def draw_labelbox(image, text, textcolor=None, labelcolor=None,
    font='simplex', fontsize=0.9, fontthickness=2, **boxargs):
    """
    Draw a labelled box in an image

    :param text: text label
    :type text: str
    :param textcolor: text color, defaults to black
    :type textcolor: str, array_like(3), optional
    :param labelcolor: label background color
    :type labelcolor: str, array_like(3), optional
    :param font: OpenCV font, defaults to cv.FONT_HERSHEY_SIMPLEX
    :type font: str, optional
    :param fontsize: OpenCV font scale, defaults to 0.3
    :type fontsize: float, optional
    :param fontthickness: font thickness in pixels, defaults to 2
    :type fontthickness: int, optional
    :param boxargs: arguments passed to :func:`draw_box`
    :raises TypeError: can't draw color into a greyscale image
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    The position of the box is specified using the same arguments as for
    :func:`draw_box`. The label font is specified using the same arguments as
    for :func:`draw_text`. If ``labelcolor`` is specified it is used as the
    background color for the text label, otherwise the box color is used.

    Example::

        >>> from machinevisiontoolbox import draw_labelbox, idisp
        >>> import numpy as np
        >>> img = np.zeros((500, 500))
        >>> draw_labelbox(img, 'labelled box', bbox=[100, 500, 300, 600],
                textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_labelbox, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_labelbox(img, 'labelled box', bbox=[100, 500, 300, 600], textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
        idisp(img)

    .. note::
        - if ``image`` is a 3-plane image then ``color`` should be a 3-vector
          or colorname string and the corresponding elements are used in 
          each plane. 

    :seealso: :func:`draw_box`, :func:`draw_text`
    """

    # get size of text:  ((w,h), baseline)
    twh = cv.getTextSize(text, _fontdict[font], fontsize, fontthickness)

    # draw the box
    bl, tr = draw_box(image, **boxargs)

    # a bit of margin, 1/2 the text height
    h = round(twh[0][1] / 2)
    h2 = round(twh[0][1] / 4)

    # draw background of the label
    if labelcolor is None:
        labelcolor = boxargs.get('color')
    draw_box(image, lt=bl, wh=(twh[0][0] + h, twh[0][1] + h), color=labelcolor, thickness=-1)

    # draw the text over the background
    draw_text(image, (bl[0] + h2, bl[1] - h2), text, color=textcolor,
        font=font, fontsize=fontsize, fontthickness=fontthickness)
    return image

def draw_text(image, pos, text=None, color=None, font='simplex', fontsize=0.3, fontthickness=2):
    """
    Draw text in image

    :param image: image to draw into, greyscale or color
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param pos: position of text (u,v)
    :type pos: array_like(2)
    :param text: text
    :type text: str
    :param color: color of text
    :type color: scalar, array_like(3), str
    :param font: font name, defaults to "simplex"
    :type font: str, optional
    :param fontsize: OpenCV font scale, defaults to 0.3
    :type fontsize: float, optional
    :param fontthickness: font thickness in pixels, defaults to 2
    :type fontthickness: int, optional
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    The position corresponds to the bottom-left corner of the text box as seen
    in the image.  The font is specified by a string which selects a Hershey
    vector (stroke) font.

    ====================  =============================================
    Font name             OpenCV font name
    ====================  =============================================
    ``'simplex'``         Hershey Roman simplex
    ``'plain'``           Hershey Roman plain
    ``'duplex'``          Hershey Roman duplex (double stroke) 
    ``'complex'``         Hershey Roman complex 
    ``'triplex'``         Hershey Romantriplex
    ``'complex-small'``   Hershey Roman complex (small)
    ``'script-simplex'``  Hershey script
    ``'script-complex'``  Hershey script complex
    ``'italic'``          Hershey italic   
    ====================  =============================================

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import draw_text, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_text(img, (100, 150), 'hello world!', color=200, fontsize=2)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_text, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_text(img, (100, 150), 'hello world!', color=200, fontsize=2)
        idisp(img)

    .. note::
        - if ``image`` is a 3-plane image then ``color`` should be a 3-vector
          or colorname string and the corresponding elements are used in 
          each plane. 

    :seealso: :func:`~spatialmath.base.graphics.plot_text` `opencv.putText <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576>`_
    """

    if not isinstance(color, int) and len(image.shape) == 2:
        raise TypeError("can't draw color into a greyscale image")

    if isinstance(color, str):
        color = color_bgr(color)

    cv.putText(image, text, pos, _fontdict[font], fontsize, color, fontthickness)
    return image

def draw_point(image, pos, marker='+', text=None, color=None, font='simplex', fontsize=0.3, fontthickness=2):
    r"""
    Draw a marker in image

    :param image: image to draw into, greyscale or color
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param pos: position of marker
    :type pos: array_like(2), ndarray(2,n), list of 2-tuples
    :param marker: marker character, defaults to '+'
    :type marker: str, optional
    :param text: text label, defaults to None
    :type text: str, optional
    :param color: text color, defaults to None
    :type color: str or array_like(3), optional
    :param font: OpenCV font, defaults to cv.FONT_HERSHEY_SIMPLEX
    :type font: str, optional
    :param fontsize: OpenCV font scale, defaults to 0.3
    :type fontsize: float, optional
    :param fontthickness: font thickness in pixels, defaults to 2
    :type fontthickness: int, optional
    :raises TypeError: can't draw color into a greyscale image
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    The ``text`` label is placed to the right of the marker, and vertically centred.
    The color of the marker can be different to the color of the text, the
    marker color is specified by a single letter in the marker string, eg. 'b+'.

    Multiple points can be marked if ``pos`` is a :math:`2 \times n` array or a list of 
    coordinate pairs.  In this case:
    
    * if ``text`` is a string it is processed with ``text.format(i)`` where ``i`` is
      the point index (starting at zero).  "{0}" within text will be substituted
      by the point index.
    * if ``text`` is a list, its elements are used to label the points

    The font is specified by a string which selects a Hershey
    vector (stroke) font.

    ====================  =============================================
    Font name             OpenCV font name
    ====================  =============================================
    ``'simplex'``         Hershey Roman simplex
    ``'plain'``           Hershey Roman plain
    ``'duplex'``          Hershey Roman duplex (double stroke) 
    ``'complex'``         Hershey Roman complex 
    ``'triplex'``         Hershey Romantriplex
    ``'complex-small'``   Hershey Roman complex (small)
    ``'script-simplex'``  Hershey script
    ``'script-complex'``  Hershey script complex
    ``'italic'``          Hershey italic   
    ====================  =============================================

    Example::

        >>> from machinevisiontoolbox import draw_point, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_point(img, (100, 300), '*', fontsize=1, color=200)
        >>> draw_point(img, (500, 300), '*', 'labelled point', fontsize=1, color=200)
        >>> draw_point(img, np.random.randint(1000, size=(2,10)), '+', 'point {0}', 100, fontsize=0.8)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_point, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_point(img, (100, 300), '*', fontsize=1, color=200)
        draw_point(img, (500, 300), '*', 'labelled point', fontsize=1, color=200)
        draw_point(img, np.random.randint(1000, size=(2,10)), '+', 'point {0}', 100, fontsize=0.8)
        idisp(img)

    .. note::
        - if ``image`` is a 3-plane image then ``color`` should be a 3-vector
          or colorname string and the corresponding elements are used in 
          each plane. 

    :seealso: :func:`~spatialmath.base.graphics.plot_point` `opencv.putText <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576>`_
    """
    fontdict = {
        'simplex': cv.FONT_HERSHEY_SIMPLEX, 
        'plain': cv.FONT_HERSHEY_PLAIN, 
        'duplex': cv.FONT_HERSHEY_DUPLEX, 
        'complex': cv.FONT_HERSHEY_COMPLEX, 
        'triplex': cv.FONT_HERSHEY_TRIPLEX, 
        'complex-small': cv.FONT_HERSHEY_COMPLEX_SMALL, 
        'script-simplex': cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 
        'script-complex': cv.FONT_HERSHEY_SCRIPT_COMPLEX, 
        'italic': cv.FONT_ITALIC,         
    }
    if not isinstance(color, int) and len(image.shape) == 2:
        raise TypeError("can't draw color into a greyscale image")
    
    if isinstance(pos, np.ndarray) and pos.shape[0] == 2:
        x = pos[0,:]
        y = pos[1,:]
    elif isinstance(pos, (tuple, list)):
        if smb.islistof(pos, (tuple, list)):
            x = [z[0] for z in pos]
            y = [z[1] for z in pos]
        else:
            x = [pos[0]]
            y = [pos[1]]

    if isinstance(color, str):
        color = color_bgr(color)

    for i, xy in enumerate(zip(x, y)):
        if isinstance(text, str):
            label = text.format(i)
        elif isinstance(text, Iterable):
            label = text[i]
        else:
            label = ''
        
        cv.putText(image, f"{marker} {label}", xy, fontdict[font], fontsize, color, fontthickness)
    return image

def draw_line(image, start, end, color, thickness=1):
    """
    Draw line in image

    :param image: image to draw into, greyscale or color
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param start: start coordinate (u,v)
    :type start: array_like(2) int
    :param end: end coordinate (u,v)
    :type end: array_like(2) int
    :param color: color of line
    :type color: scalar, array_like(3)
    :param thickness: width of line in pixels, defaults to 1
    :type thickness: int, optional
    :raises TypeError: can't draw color into a greyscale image
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import draw_line, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_line(img, (100, 300), (700, 900), color=200, thickness=10)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_line, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_line(img, (100, 300), (700, 900), color=200, thickness=10)
        idisp(img)

    .. note::
        - if ``image`` is a 3-plane image then ``color`` should be a 3-vector
          or colorname string and the corresponding elements are used in 
          each plane. 

    :seealso: :func:`~spatialmath.base.graphics.plot_line` `opencv.line <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2>`_
    """
    if not isinstance(color, int) and len(image.shape) == 2:
        raise TypeError("can't draw color into a greyscale image")
    cv.line(image, start, end, color, thickness)
    return image

def draw_circle(image, centre, radius, color, thickness=1):
    """
    Draw line in image

    :param image: image to draw into, greyscale or color
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param centre: centre coordinate
    :type start: array_like(2) int
    :param radius: radius in pixels
    :type end: int
    :param color: color of circle
    :type color: scalar, array_like(3)
    :param thickness: width of line in pixels, -1 to fill, defaults to 1
    :type thickness: int, optional
    :raises TypeError: can't draw color into a greyscale image
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    Example::

        >>> from machinevisiontoolbox import draw_circle, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_circle(img, (400,600), 150, thickness=2, color=200)
        >>> draw_circle(img, (400,600), 150, thickness=-1, color=50)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_circle, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_circle(img, (400,600), 150, thickness=2, color=200)
        draw_circle(img, (400,600), 150, thickness=-1, color=50)
        idisp(img)

    .. note::
        - if ``image`` is a 3-plane image then ``color`` should be a 3-vector
          or colorname string and the corresponding elements are used in 
          each plane. 

    :seealso: :func:`~spatialmath.base.graphics.plot_circle` `opencv.circle <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670>`_
    """
    if not isinstance(color, int) and len(image.shape) == 2:
        raise TypeError("can't draw color into a greyscale image")
    cv.circle(image, centre, radius, color, thickness)
    return image

# def plot_histogram(c, n, clip=False, ax=None, block=False, xlabel=None, ylabel=None, grid=False, **kwargs):
#     if ax is None:
#         plt.figure()
#         ax = plt.gca()

#     # n = hist.h  # number of pixels per class
#     # c = hist.x  # class value

#     if clip:
#         nz, _ = np.where(n > 0)
#         start = nz[0]
#         end = nz[-1] + 1
#         n = n[start:end]
#         c = c[start:end]

#     ax.bar(c, n, **kwargs)
#     if xlabel is not None:
#         ax.set_xlabel(xlabel)
#     if ylabel is not None:
#         ax.set_ylabel(ylabel)
#     ax.grid(grid)

    # plt.show(block=block)

# if __name__ == "__main__":

#     import numpy as np
#     from machinevisiontoolbox import idisp, iread, Image

#     from machinevisiontoolbox import draw_labelbox
#     import numpy as np
#     img = np.zeros((1000, 1000), dtype='uint8')
#     draw_labelbox(img, 'labelled box', bbox=[100, 500, 300, 600],
#         textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
#     idisp(img, block=True)

#     im = np.zeros((100,100,3), 'uint8')
#     im, file = iread('flowers1.png')

#     draw_box(im, color=(255,0,0), centre=(50,50), wh=(20,20))

#     draw_point(im, [(200,200), (300, 300), (400,400)], color='blue')

#     draw_labelbox(im, "box", thickness=3, centre=(100,100), wh=(100,30), color='red', textcolor='white')
#     idisp(im, block=True)

    