"""
Shared interactive rectangle / line-profile overlay for MVTB image display tools.

Provides :class:`ImageInteractor`, which wires keyboard and mouse callbacks for
rectangle selection, edge-handle dragging, intensity line profiling, and ROI
histogram display onto any matplotlib figure that is showing an
:class:`~machinevisiontoolbox.Image`.
"""

import faulthandler
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

# Enable C-level crash tracebacks (SIGSEGV, SIGBUS, etc.)
faulthandler.enable()


def _guard_popup_close(fig) -> None:
    """
    Defuse pending draw_idle timers when a popup figure window is closed.

    macOS-specific race: ``draw_idle()`` schedules a one-shot NSTimer whose
    Python callback is ``_draw_idle()``.  ``_draw_idle()`` gates itself with
    the ``_draw_pending`` flag: when the flag is *False* it returns immediately
    without touching the C-level canvas.  Setting the flag to *False* inside
    the ``close_event`` handler (which fires while the canvas is still fully
    alive) is therefore enough to neutralise any timer that is still in flight
    after the window is destroyed -- the timer fires, ``_draw_idle`` sees
    ``_draw_pending=False`` and exits without calling ``draw()``, so no
    CoreGraphics access occurs on a freed canvas.

    This is backend-agnostic: on Qt/Tk/GTK ``_draw_pending`` is either absent
    or ``False`` already, so the assignment is a harmless no-op.
    """

    def _on_close(event):
        event.canvas._draw_pending = False  # defuse pending draw_idle timer

    fig.canvas.mpl_connect("close_event", _on_close)


# Named colours for standard RGBA image planes.
_PLANE_COLORS: dict[str, str] = {
    "R": "red",
    "G": "yellowgreen",
    "B": "blue",
    "A": "black",
}

_HELP_HEADER = (
    "\n"
    "Image interactive shortcuts\n"
    "───────────────────────────────────────\n"
    "  left-drag            draw a selection rectangle\n"
    "    (only when toolbar zoom/pan is inactive)\n"
    "  drag handle          resize rectangle edge\n"
    "  shift + left-drag    draw a line for intensity profile\n"
    "\n"
    "  h   histogram of selection\n"
    "  c   cumulative histogram of selection\n"
    "  p   pop selection out into new window\n"
    "\n"
    "  ?   show this help\n"
    "\n"
    "Note: popup windows are unstable with the macOS backend.\n"
    "      Use --backend Qt5Agg for reliable operation.\n"
)


class ImageInteractor:
    """
    Attach interactive rectangle / line-profile callbacks to a matplotlib
    figure currently displaying a :class:`~machinevisiontoolbox.Image`.

    :param fig: matplotlib figure
    :param ax: axes within *fig* that display the image
    :param image: the Image currently displayed
    :param nav_help: optional extra help text appended to the standard
        shortcut table when the user presses ``?``

    After construction the interactor is immediately live.  Call
    :meth:`update_image` whenever the displayed frame changes (e.g. when
    stepping through a bag file) to reset selection state.  Call
    :meth:`detach` to disconnect all callbacks.
    """

    def __init__(self, fig, ax, image, nav_help: str = "") -> None:
        self._fig = fig
        self._ax = ax
        self._image = image
        self._nav_help = nav_help

        self._s: dict = {
            "rect": None,  # [u0, v0, u1, v1] raw drag coords
            "rect_patch": None,  # matplotlib Rectangle patch
            "handles": {},  # edge-name -> Circle patch
            "drag_start": None,  # (u, v) on press
            "drag_mode": None,  # 'rect' | 'line' | 'handle' | None
            "active_handle": None,  # 'left' | 'right' | 'top' | 'bottom'
            "line_patch": None,  # Line2D for live line preview
            "line_start": None,  # (u, v) of line press
        }

        self._alt_down: bool = False  # tracked via key press/release events

        # Strong references to popup figures.  matplotlib's Gcf drops its
        # reference when a popup window is closed, which frees the C-level
        # FigureCanvas (NSView).  Any pending draw_idle NSTimer then fires on
        # freed memory → SIGSEGV.  Keeping the Figure alive here keeps the
        # NSView alive until detach() so the timer fires safely on a
        # detached-but-valid view.
        self._popup_figs: list = []

        self._cids = [
            fig.canvas.mpl_connect("button_press_event", self._on_press),
            fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            fig.canvas.mpl_connect("button_release_event", self._on_release),
            fig.canvas.mpl_connect("key_press_event", self._on_key),
            fig.canvas.mpl_connect("key_release_event", self._on_key_release),
        ]

        # Keep a strong reference on the figure so the interactor is not
        # garbage-collected while the figure is open (matplotlib uses
        # WeakMethod internally for bound-method callbacks).
        fig._iminteractor = self

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_image(self, image) -> None:
        """Reset selection state and update the image reference for a new frame."""
        self._image = image
        if self._s["rect"] is not None:
            self._clear_selection()
        if self._s["line_patch"] is not None:
            self._s["line_patch"].remove()
            self._s["line_patch"] = None
        self._s["drag_mode"] = None
        self._s["drag_start"] = None
        self._s["line_start"] = None
        self._fig.canvas.draw_idle()

    def detach(self) -> None:
        """Disconnect all event callbacks from the figure."""
        for cid in self._cids:
            self._fig.canvas.mpl_disconnect(cid)
        self._cids.clear()
        if getattr(self._fig, "_iminteractor", None) is self:
            del self._fig._iminteractor
        self._popup_figs.clear()  # allow GC once event loop has stopped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _handle_r(self) -> float:
        return max(self._image.width, self._image.height) * 0.010

    def _toolbar_active(self) -> bool:
        """Return True when the MPL toolbar has zoom or pan active."""
        try:
            mode = str(self._fig.canvas.manager.toolbar.mode)
            return bool(mode)
        except Exception:
            return False

    def _norm_rect(self):
        u0, v0, u1, v1 = self._s["rect"]
        return min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)

    def _update_rect_patch(self) -> None:
        u0, v0, u1, v1 = self._norm_rect()
        p = self._s["rect_patch"]
        if p is None:
            p = Rectangle(
                (u0, v0),
                u1 - u0,
                v1 - v0,
                linewidth=1.5,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
            )
            self._ax.add_patch(p)
            self._s["rect_patch"] = p
        else:
            p.set_xy((u0, v0))
            p.set_width(u1 - u0)
            p.set_height(v1 - v0)

    def _update_handles(self) -> None:
        u0, v0, u1, v1 = self._norm_rect()
        cx, cy = (u0 + u1) / 2, (v0 + v1) / 2
        positions = {
            "left": (u0, cy),
            "right": (u1, cy),
            "top": (cx, v0),
            "bottom": (cx, v1),
        }
        r = self._handle_r
        for hname, pos in positions.items():
            if hname in self._s["handles"]:
                self._s["handles"][hname].center = pos
            else:
                c = Circle(
                    pos,
                    r,
                    edgecolor="yellow",
                    facecolor="white",
                    linewidth=1.5,
                    zorder=5,
                )
                self._ax.add_patch(c)
                self._s["handles"][hname] = c

    def _clear_selection(self) -> None:
        if self._s["rect_patch"] is not None:
            self._s["rect_patch"].remove()
            self._s["rect_patch"] = None
        for c in self._s["handles"].values():
            c.remove()
        self._s["handles"].clear()
        self._s["rect"] = None

    def _hit_handle(self, u: float, v: float):
        r = self._handle_r
        for hname, c in self._s["handles"].items():
            cx, cy = c.center
            if (u - cx) ** 2 + (v - cy) ** 2 <= r**2:
                return hname
        return None

    def _roi_image(self):
        image = self._image
        u0, v0, u1, v1 = self._norm_rect()
        u0 = max(0, min(int(round(u0)), image.width - 1))
        v0 = max(0, min(int(round(v0)), image.height - 1))
        u1 = max(0, min(int(round(u1)), image.width - 1))
        v1 = max(0, min(int(round(v1)), image.height - 1))
        return image.roi([u0, u1, v0, v1])

    def _show_profile(self, u0: float, v0: float, u1: float, v1: float) -> None:
        image = self._image
        length = max(2, int(round(np.hypot(u1 - u0, v1 - v0))))
        us = np.clip(np.linspace(u0, u1, length), 0, image.width - 1)
        vs = np.clip(np.linspace(v0, v1, length), 0, image.height - 1)
        ui = np.round(us).astype(int)
        vi = np.round(vs).astype(int)

        arr = image.array
        pfig, pax = plt.subplots()
        if pfig.canvas.manager is not None:
            pfig.canvas.manager.set_window_title("Intensity profile")

        if arr.ndim == 2:
            pax.plot(arr[vi, ui])
        else:
            names = (image.colororder_str or "").split(":")
            for p in range(arr.shape[2]):
                label = names[p] if p < len(names) else f"plane {p}"
                color = _PLANE_COLORS.get(label)
                pax.plot(arr[vi, ui, p], label=label, color=color)
            pax.legend()

        pax.set_xlabel("Distance along line (pixels)")
        pax.set_ylabel("Intensity")
        pax.set_xlim(0, length - 1)
        pax.grid(True)
        pax.set_title(
            f"Profile ({u0:.0f},\u202f{v0:.0f}) \u2192 ({u1:.0f},\u202f{v1:.0f})"
        )
        pfig.canvas.draw_idle()  # safe from event callbacks – just schedules NSTimer
        self._popup_figs.append(pfig)  # keep Python Figure alive (timer closure guard)
        _guard_popup_close(pfig)
        pfig.show()

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax or event.button != 1:
            return
        u, v = event.xdata, event.ydata
        if u is None or v is None:
            return

        if event.key == "shift":
            self._s["drag_mode"] = "line"
            self._s["line_start"] = (u, v)
            if self._s["line_patch"] is not None:
                self._s["line_patch"].remove()
                self._s["line_patch"] = None
        elif not self._toolbar_active():
            hit = self._hit_handle(u, v) if self._s["rect"] is not None else None
            if hit:
                self._s["drag_mode"] = "handle"
                self._s["active_handle"] = hit
            else:
                self._clear_selection()
                self._s["drag_mode"] = "rect"
                self._s["drag_start"] = (u, v)

    def _on_motion(self, event) -> None:
        if event.inaxes is not self._ax or event.button != 1:
            return
        u, v = event.xdata, event.ydata
        if u is None or v is None:
            return

        if self._s["drag_mode"] == "rect":
            u0, v0 = self._s["drag_start"]
            self._s["rect"] = [u0, v0, u, v]
            self._update_rect_patch()
            self._fig.canvas.draw_idle()

        elif self._s["drag_mode"] == "line":
            u0, v0 = self._s["line_start"]
            if self._s["line_patch"] is None:
                (line,) = self._ax.plot(
                    [u0, u],
                    [v0, v],
                    color="yellow",
                    linestyle="--",
                    linewidth=1.5,
                    zorder=5,
                )
                self._s["line_patch"] = line
            else:
                self._s["line_patch"].set_data([u0, u], [v0, v])
            self._fig.canvas.draw_idle()

        elif self._s["drag_mode"] == "handle":
            ha = self._s["active_handle"]
            if ha == "left":
                self._s["rect"][0] = u
            elif ha == "right":
                self._s["rect"][2] = u
            elif ha == "top":
                self._s["rect"][1] = v
            elif ha == "bottom":
                self._s["rect"][3] = v
            self._update_rect_patch()
            self._update_handles()
            self._fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if event.button != 1:
            return
        mode = self._s["drag_mode"]
        self._s["drag_mode"] = None

        if mode == "rect" and self._s["rect"] is not None:
            u0, v0, u1, v1 = self._norm_rect()
            if (u1 - u0) > 3 and (v1 - v0) > 3:
                self._update_handles()
            else:
                self._clear_selection()
            self._fig.canvas.draw_idle()

        elif mode == "line":
            u, v = event.xdata, event.ydata
            if self._s["line_patch"] is not None and u is not None and v is not None:
                u0, v0 = self._s["line_start"]
                if np.hypot(u - u0, v - v0) > 3:
                    self._show_profile(u0, v0, u, v)
            # Always clear the guide line on release so a subsequent
            # plain left-drag creates a fresh rectangle rather than
            # updating the old line.
            if self._s["line_patch"] is not None:
                self._s["line_patch"].remove()
                self._s["line_patch"] = None
            self._s["line_start"] = None
            self._fig.canvas.draw_idle()

        elif mode == "handle":
            self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ("alt", "Alt_L", "Alt_R"):
            self._alt_down = True
            return

        if event.key == "?":
            msg = _HELP_HEADER
            if self._nav_help:
                msg += self._nav_help
            print(msg)
            return

        if self._s["rect"] is not None:
            roi = self._roi_image()
            # strip modifier prefix: macOS reports 'alt+h', 'ctrl+h', etc.
            base_key = event.key.split("+")[-1] if event.key else ""
            if base_key == "h":
                self._popup_hist(roi, cumulative=False)
            elif base_key == "c":
                self._popup_hist(roi, cumulative=True)
            elif base_key == "p":
                self._popup_roi(roi)

    def _defer(self, fn) -> None:
        """Run *fn* after the current event callback returns via a one-shot timer."""
        timer = self._fig.canvas.new_timer(interval=0)

        def _fire():
            try:
                fn()
            except Exception:
                import traceback

                traceback.print_exc()
            finally:
                timer.stop()

        timer.add_callback(_fire)
        timer.start()

    def _popup_hist(self, roi, *, cumulative: bool) -> None:
        try:
            label = "Cumulative histogram" if cumulative else "Histogram"
            title = f"{label} \u2013 ROI"
            kwargs = dict(block=None, cursor=True, title=title)
            if cumulative:
                kwargs["type"] = "cdf"
            roi.hist().plot(**kwargs)
            fig = plt.gcf()
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(title)
            fig.canvas.draw_idle()  # safe from event callbacks – just schedules NSTimer
            self._popup_figs.append(
                fig
            )  # keep Python Figure alive (timer closure guard)
            _guard_popup_close(fig)
            fig.show()
        except Exception:
            import traceback

            traceback.print_exc()

    def _popup_roi(self, roi) -> None:
        try:
            roi.disp(block=None, title="ROI")
            fig = plt.gcf()
            fig.canvas.draw_idle()  # safe from event callbacks – just schedules NSTimer
            self._popup_figs.append(
                fig
            )  # keep Python Figure alive (timer closure guard)
            _guard_popup_close(fig)
            fig.show()
        except Exception:
            import traceback

            traceback.print_exc()

    def _on_key_release(self, event) -> None:
        if event.key in ("alt", "Alt_L", "Alt_R"):
            self._alt_down = False
