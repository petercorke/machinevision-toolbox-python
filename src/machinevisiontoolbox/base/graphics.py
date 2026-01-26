import cv2 as cv
from ansitable import ANSITable, Column
from machinevisiontoolbox.base import name2color
import matplotlib.pyplot as plt
import numpy as np
import spatialmath.base as smb
from collections.abc import Iterable


def _color(image, color):
    if color is None:
        return None
    if isinstance(color, str):
        color = name2color(color, dtype=image.dtype)
        if np.issubdtype(image.dtype, np.integer):
            # OpenCV wants a tuple of Python ints
            color = tuple([int(c) for c in color])

    if isinstance(color, int):
        if image.ndim > 2:
            # integer color for multiplane image
            color = [color] * image.shape[2]
    else:
        if len(color) > 1 and image.ndim == 2:
            raise ValueError(
                f"color has multiple elements ({len(color)}), image has only one plane"
            )
        elif len(color) != image.shape[2]:
            raise ValueError(
                f"number of elements of color ({len(color)} differs from number of image planes ({image.shape[2]})"
            )
    if not isinstance(color, int):
        return tuple(color)
    else:
        return color


def _roundvec(x):
    """Round an iterable or ndarray to a tuple of integers

    :param x: iterable or ndarray to round
    :type x: array_like(2), 2-element list or tuples
    :return: rounded vector
    :rtype: tuple
    """
    if isinstance(x, np.ndarray):
        x = x.round(0).flatten().astype(int)
    else:
        x = map(lambda y: round(y), x)
    return tuple(x)


def draw_box(
    image,
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
    lbrt=None,
    lrbt=None,
    ltrb=None,
    lbwh=None,
    ax=None,
    color=None,
    thickness=1,
    antialias=False,
):
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

    :param lb: left-bottom corner [u,v]
    :type lb: array_like(2), optional
    :param lt: left-top corner [u,v]
    :type lt: array_like(2), optional
    :param rb: right-bottom corner (u,v)
    :type rb: array_like(2), optional
    :param rt: right-top corner (u,v)
    :type rt: array_like(2), optional
    :param wh: width and height
    :type wh: array_like(2), optional
    :param centre: box centre (u,v)
    :type centre: array_like(2), optional

    :param lbrt: left-bottom-right-top [xmin, ymin, xmax, ymax]
    :type lbrt: array_like(4), optional
    :param lrbt: left-right-bottom-top [xmin, ymin, xmax, ymax]
    :type lrbt: array_like(4), optional
    :param ltrb: bounding box [xmin, ymin, xmax, ymax]
    :type ltrb: array_like(4), optional
    :param lbwh: left-bottom-width-height [xmin, ymin, width, height]
    :type lbwh: array_like(4), optional

    :param color: color of line
    :type color: scalar or array_like
    :param thickness: line thickness, -1 to fill, defaults to 1
    :type thickness: int, optional
    :param antialias: use antialiasing, defaults to False
    :type antialias: bool, optional
    :param ax: axes to draw into
    :type: Matplotlib axes

    :return: bottom-left and top-right corners of the box
    :rtype: (2-tuple, 2-tuple)

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
        >>> draw_box(img, lbrt=[100, 300, 700, 500], thickness=2, color=200) # outline box
        >>> draw_box(img, lbwh=[300, 400, 500, 400], thickness=-1, color=250) # filled box
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_box, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_box(img, lbrt=[100, 300, 700, 500], thickness=2, color=200)
        draw_box(img, lbwh=[300, 400, 500, 400], thickness=-1, color=250)
        idisp(img)

    .. warning:: For images y increases downwards so top of the box, has a larger
        v-coordinate, and is lower in the image.

    .. note:: If ``image`` has multiple planes then ``color`` should have the same number
        of elements as the image has planes. If it is a scalar that value is used
        for each color plane. For a color image ``color`` can be
        a string color name.

    :seealso: :func:`~smtb.base.graphics.plot_box`  `opencv.rectangle <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9>`_
    """

    # test for various 4-coordinate versions
    if lbwh is not None:
        l, b = lbwh[:2]
        w, h = lbwh[2:]

    elif lbrt is not None:
        l, b = lbrt[:2]
        r, t = lbrt[2:]

    elif lrbt is not None:
        l, b = (lrbt[0], lrbt[2])
        r, t = (lrbt[1], lrbt[3])

    elif ltrb is not None:
        l, b = (ltrb[0], ltrb[3])
        r, t = (ltrb[2], ltrb[1])

    # test for 2-vectors for corners
    if lb is not None:
        l, b = lb

    if lt is not None:
        l, t = lt

    if rb is not None:
        r, b = rb

    if rt is not None:
        r, t = rt

    if wh is not None:
        if smb.isscalar(wh):
            w, h = wh, wh
        else:
            w, h = wh

    # at this point we have some of: l, r, b, t, w, h

    try:
        if w is not None and h is not None:
            # we have width & height, one corner is enough

            if centre is not None:
                l, b = (centre[0] - w / 2, centre[1] - h / 2)
                r, t = (centre[0] + w / 2, centre[1] + h / 2)

            else:
                if r is None:
                    r = l + w
                if t is None:
                    t = b + h
                if l is None:
                    l = r - w
                if b is None:
                    b = t - h

        if l > r:
            raise ValueError("left must be less than right")
        if b > t:
            raise ValueError("bottom must be less than top")

    except TypeError:
        raise ValueError("insufficent parameters to compute a box")

    color = _color(image, color)

    if antialias:
        linetype = cv.LINE_AA
    else:
        linetype = cv.LINE_8
    cv.rectangle(image, lb := (l, b), rt := (r, t), color, thickness, linetype)

    return lb, rt


def plot_labelbox(text, textcolor=None, labelcolor=None, position="topleft", **boxargs):
    """
    Plot a labelled box using Matplotlib

    :param text: text label
    :type text: str
    :param textcolor: text color, defaults to None
    :type textcolor: str, array_like(3), optional
    :param labelcolor: label background color
    :type labelcolor: str, array_like(3), optional
    :param position: place to draw the label: 'topleft' (default), 'topright, 'bottomleft' or 'bottomright'
    :type above: str, optional
    :param boxargs: arguments passed to :func:`plot_box`

    Plot a box with a label above it. The position of the box is specified using
    the same arguments as for ``plot_box``. The label font is specified using
    the same arguments as for ``plot_text``.  If ``labelcolor`` is specified it
    is used as the background color for the text label, otherwise the box color
    is used.

    Example::

        >>> from machinevisiontoolbox import plot_labelbox
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> idisp(img) # create a Matplotlib window
        >>> plot_labelbox("labelled box", lbwh=[100, 250, 300, 400], color="yellow")
        >>> plot_labelbox('another labelled box', position="bottomright", lbwh=[300, 450, 500, 400], color="red")

    .. plot::

        from machinevisiontoolbox import plot_labelbox
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        idisp(img)
        plot_labelbox('labelled box', lbwh=[100, 250, 300, 400], color="yellow")
        plot_labelbox('another labelled box', position="bottomright", lbwh=[300, 450, 400, 400], color="red")



    :seealso: :func:`~spatialmath.base.plot_box`, :func:`~spatialmath.base.plot_text`
    """

    rect = smb.plot_box(**boxargs)

    bbox = rect.get_bbox()

    if labelcolor is None:
        labelcolor = boxargs.get("color")

    if position == "topleft":
        pos = (bbox.xmin, bbox.ymin)
        valign = "bottom"
        halign = "left"
    elif position == "topright":
        pos = (bbox.xmax, bbox.ymin)
        valign = "bottom"
        halign = "right"
    elif position == "bottomleft":
        pos = (bbox.xmin, bbox.ymax)
        valign = "top"
        halign = "left"
    elif position == "bottomright":
        pos = (bbox.xmax, bbox.ymax)
        valign = "top"
        halign = "right"

    smb.plot_text(
        pos,
        text,
        color=textcolor,
        verticalalignment=valign,
        horizontalalignment=halign,
        bbox=dict(facecolor=labelcolor, linewidth=0, edgecolor=None),
    )

    return rect


_fontdict = {
    "simplex": cv.FONT_HERSHEY_SIMPLEX,
    "plain": cv.FONT_HERSHEY_PLAIN,
    "duplex": cv.FONT_HERSHEY_DUPLEX,
    "complex": cv.FONT_HERSHEY_COMPLEX,
    "triplex": cv.FONT_HERSHEY_TRIPLEX,
    "complex-small": cv.FONT_HERSHEY_COMPLEX_SMALL,
    "script-simplex": cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script-complex": cv.FONT_HERSHEY_SCRIPT_COMPLEX,
    "italic": cv.FONT_ITALIC,
}


def draw_labelbox(
    image,
    text,
    textcolor=None,
    labelcolor=None,
    font="simplex",
    fontsize=0.9,
    fontheight=None,
    fontthickness=2,
    position="topleft",
    **boxargs,
):
    """
    Draw a labelled box in an image

    :param text: text label
    :type text: str
    :param textcolor: text color, defaults to black
    :type textcolor: str, array_like(3), optional
    :param labelcolor: label background color
    :type labelcolor: str, array_like(3), optional
    :param position: place to draw the label: 'topleft' (default), 'topright, 'bottomleft' or 'bottomright'
    :type above: str, optional
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
        >>> draw_labelbox(img, "labelled box", lbwh=[100, 200, 400, 500], textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
        >>> draw_labelbox(img, "another labelled box", position="bottomright", lbwh=[300, 450, 500, 400], textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_labelbox, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_labelbox(img, "labelled box", lbwh=[100, 200, 400, 500], textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
        draw_labelbox(img, "another labelled box", position="bottomright", lbwh=[300, 450, 400, 400], textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
        idisp(img)

    .. note:: If ``image`` has multiple planes then ``color``, ``labelcolor`` and
        ``textcolor`` should have the same number
        of elements as the image has planes. If they are a scalar that value is used
        for each color plane. For a color image ``color`` can be
        a string color name.

    :seealso: :func:`draw_box`, :func:`draw_text`
    """

    if fontheight is not None:
        fontsize = cv.getFontScaleFromHeight(_fontdict[font], fontheight, fontthickness)

    # get size of text:  ((w,h), baseline)
    w, h = cv.getTextSize(text, _fontdict[font], fontsize, fontthickness)[0]

    # draw the box
    lb, rt = draw_box(image, **boxargs)

    # a bit of margin, 1/2 the text height
    h2 = round(h / 2)
    h4 = round(h / 4)

    # draw background of the label
    if labelcolor is None:
        labelcolor = boxargs.get("color")

    # draw the text over the background
    if position == "topleft":
        pos = (lb[0], lb[1])
    elif position == "topright":
        pos = (rt[0] - w - h4, lb[1])
    elif position == "bottomleft":
        pos = (lb[0], rt[1] + h + h2)
    elif position == "bottomright":
        pos = (rt[0] - w - h4, rt[1] + h + h2)

    # draw the label background
    draw_box(image, lt=pos, wh=(w + h2, h + h2), color=labelcolor, thickness=-1)

    # draw the label text
    draw_text(
        image,
        (pos[0] + h4, pos[1] - h4),
        text,
        color=textcolor,
        font=font,
        fontsize=fontsize,
        fontthickness=fontthickness,
    )
    return image


def draw_text(
    image,
    pos,
    text=None,
    color=None,
    font="simplex",
    fontheight=None,
    fontsize=0.3,
    fontthickness=2,
    antialias=False,
):
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
    :param fontheight: height of font in pixels, defaults to None
    :type fontheight: int, optional
    :param fontsize: OpenCV font scale, defaults to 0.3
    :type fontsize: float, optional
    :param fontthickness: font thickness in pixels, defaults to 2
    :type fontthickness: int, optional
    :param antialias: use antialiasing, defaults to False
    :type antialias: bool, optional
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

    Example::

        >>> from machinevisiontoolbox import draw_text, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_text(img, (100, 150), 'Hello world!', color=200, fontheight=60)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_text, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_text(img, (100, 150), 'Hello world!', color=200, fontheight=60)
        idisp(img)

    .. note:: Font size can be specified in two ways:

        - ``fontsize`` is the OpenCV font size scale factor as used by
          :func:`opencv.putText`
        - ``fontheight`` is the height of the font in pixels, this overrides
          ``fontsize``.  The font scale is computed from ``fontheight`` using
          :func:`opencv.getFontScaleFromHeight`

    .. note:: If ``image`` has multiple planes then ``color`` should have the same number
          of elements as the image has planes. If it is a scalar that value is used
          for each color plane. For a color image ``color`` can be
          a string color name.

    :seealso: :func:`~spatialmath.base.graphics.plot_text` `opencv.putText <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576>`_
    """
    if fontheight is not None:
        fontsize = cv.getFontScaleFromHeight(_fontdict[font], fontheight, fontthickness)

    color = _color(image, color)

    if antialias:
        lt = cv.LINE_AA
    else:
        lt = cv.LINE_8
    cv.putText(
        image, text, _roundvec(pos), _fontdict[font], fontsize, color, fontthickness, lt
    )
    return image


def draw_point(
    image,
    pos,
    marker="+",
    text=None,
    color=None,
    font="simplex",
    fontheight=None,
    fontsize=0.3,
    fontthickness=2,
):
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
    :param fontheight: height of font in pixels, defaults to None
    :type fontheight: int, optional
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
    ``"simplex"``         Hershey Roman simplex
    ``"plain"``           Hershey Roman plain
    ``"duplex"``          Hershey Roman duplex (double stroke)
    ``"complex"``         Hershey Roman complex
    ``"triplex"``         Hershey Romantriplex
    ``"complex-small"``   Hershey Roman complex (small)
    ``"script-simplex"``  Hershey script
    ``"script-complex"``  Hershey script complex
    ``"italic"``          Hershey italic
    ====================  =============================================

    .. note:: Font size can be specified in two ways:

        - ``fontsize`` is the OpenCV font size scale factor as used by
          :func:`opencv.putText`
        - ``fontheight`` is the height of the font in pixels, this overrides
          ``fontsize``.  The font scale is computed from ``fontheight`` using
          :func:`opencv.getFontScaleFromHeight`

    .. note:: The centroid of the marker character is very accurately positioned at
        the specified coordinate.  The text label is placed to the right of the
        marker.

    .. note:: If ``image`` has multiple planes then ``color`` should have the same number
          of elements as the image has planes. If it is a scalar that value is used
          for each color plane. For a color image ``color`` can be
          a string color name.

    Example::

        >>> from machinevisiontoolbox import draw_point, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_point(img, (100, 300), '*', fontsize=1, color=200)
        >>> draw_point(img, (500, 300), '*', 'labelled point', fontsize=1, color=200)
        >>> draw_point(img, np.random.randint(1000, size=(2,10)), '+', 'point {0}', color=100, fontsize=0.8)
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_point, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_point(img, (100, 300), '*', fontsize=1, color=200)
        draw_point(img, (500, 300), '*', 'labelled point', fontsize=1, color=200)
        draw_point(img, np.random.randint(1000, size=(2,10)), '+', 'point {0}', color=100, fontsize=0.8)
        idisp(img)


    :seealso: :func:`~spatialmath.base.graphics.plot_point` `opencv.putText <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576>`_
    """

    if fontheight is not None:
        fontsize = cv.getFontScaleFromHeight(_fontdict[font], fontheight, fontthickness)

    if isinstance(pos, np.ndarray) and pos.shape[0] == 2:
        x = pos[0, :]
        y = pos[1, :]
    elif isinstance(pos, (tuple, list)):
        if smb.islistof(pos, (tuple, list)):
            x = [z[0] for z in pos]
            y = [z[1] for z in pos]
        else:
            x = [pos[0]]
            y = [pos[1]]

    newmarker = ""
    markercolor = ""
    for m in marker:
        if m in "rgbcmykw":
            markercolor += m
        else:
            newmarker += m
    marker = newmarker

    # get the centre of the marker, cv.getTextSize is a very loose bounding box
    #  the code below is a bit expensive but the only way to precisely position
    #  the marker
    tmp = np.zeros((200, 200), dtype="uint8")
    cv.putText(tmp, marker, (0, 150), _fontdict[font], fontsize, 1, fontthickness)
    v, u = np.argwhere(tmp > 0).T
    uc = u.mean()
    vc = v.mean() - 150

    if color is None:
        color = markercolor

    color = _color(image, color)

    for i, xy in enumerate(zip(x, y)):
        if isinstance(text, str):
            label = f"{marker} {text.format(i)}"
        elif isinstance(text, Iterable):
            label = f"{marker} {text[i]}"
        else:
            label = marker

        x = round(xy[0] - uc)
        y = round(xy[1] - vc)
        cv.putText(
            image,
            label,
            (x, y),
            _fontdict[font],
            fontsize,
            color,
            fontthickness,
        )
    return image


def draw_line(image, start, end, color, thickness=1, antialias=False):
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
    :param antialias: use antialiasing, defaults to False
    :type antialias: bool, optional
    :raises TypeError: can't draw color into a greyscale image
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    The coordinates can be tuples, lists or NumPy arrays.  The values are rounded to
    the nearest integer.

    Example::

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

    .. note:: If ``image`` has multiple planes then ``color`` should have the same number
          of elements as the image has planes. If it is a scalar that value is used
          for each color plane. For a color image ``color`` can be
          a string color name.

    :seealso: :func:`~spatialmath.base.graphics.plot_line` `opencv.line <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2>`_
    """
    color = _color(image, color)

    if antialias:
        lt = cv.LINE_AA
    else:
        lt = cv.LINE_8

    cv.line(image, _roundvec(start), _roundvec(end), color, thickness, lt)
    return image


def draw_circle(image, centre, radius, color, thickness=1, antialias=False):
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
    :param antialias: use antialiasing, defaults to False
    :type antialias: bool, optional
    :raises TypeError: can't draw color into a greyscale image
    :return: passed image as modified
    :rtype: ndarray(H,W), ndarray(H,W,P)

    The centre coordinate can be a tuple, list or NumPy array.  The values are rounded to
    the nearest integer.  The radius is also rounded to the nearest integer.

    Example::

        >>> from machinevisiontoolbox import draw_circle, idisp
        >>> import numpy as np
        >>> img = np.zeros((1000, 1000), dtype='uint8')
        >>> draw_circle(img, (300,400), 150, thickness=2, color=200)
        >>> draw_circle(img, (500,700), 250, thickness=-1, color=50)  # filled
        >>> draw_circle(img, (900,900), 200, thickness=-1, color=100)  # filled
        >>> idisp(img)

    .. plot::

        from machinevisiontoolbox import draw_circle, idisp
        import numpy as np
        img = np.zeros((1000, 1000), dtype='uint8')
        draw_circle(img, (300,400), 150, thickness=2, color=200)
        draw_circle(img, (500,700), 250, thickness=-1, color=50)
        draw_circle(img, (900,900), 200, thickness=-1, color=100)  # filled
        idisp(img)

    .. note:: If ``image`` has multiple planes then ``color`` should have the same number
          of elements as the image has planes. If it is a scalar that value is used
          for each color plane. For a color image ``color`` can be
          a string color name.

    :seealso: :func:`~spatialmath.base.graphics.plot_circle` `opencv.circle <https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670>`_
    """
    color = _color(image, color)

    if antialias:
        lt = cv.LINE_AA
    else:
        lt = cv.LINE_8
    cv.circle(image, _roundvec(centre), round(radius), color, thickness, lt)
    return image


if __name__ == "__main__":
    from machinevisiontoolbox.base import draw_box, idisp
    import numpy as np


#     import numpy as np
#     from machinevisiontoolbox import idisp, iread, Image

#     from machinevisiontoolbox import draw_labelbox

#     img = np.zeros((1000, 1000, 3), dtype="float32")
#     draw_box(img, lrbt=[100, 400, 100, 400], color="red", thickness=-1)
#     draw_box(img, lrbt=[500, 800, 100, 400], color="green", thickness=-1)
#     draw_box(img, lrbt=[100, 400, 500, 800], color="blue", thickness=-1)
#     draw_box(img, lrbt=[500, 800, 500, 800], color="white", thickness=-1)
#     idisp(img, block=True)
