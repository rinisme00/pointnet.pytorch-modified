import numpy as np
import ctypes as ct
import cv2
import sys

#
# show3d_balls.py — robust OpenGL-like point-cloud viewer fallback
# on headless servers: auto-disable GUI and renderer
#
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True

# Attempt to create a window; if unavailable, fall back gracefully
_HAS_GUI = True

def onmouse(event, x, y, flags, param):
    global mousex, mousey, changed
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True

# Try to initialize GUI
try:
    cv2.namedWindow('show3d', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('show3d', showsz, showsz)
    cv2.setMouseCallback('show3d', onmouse)
except cv2.error:
    _HAS_GUI = False
    print("show3d_balls: GUI unavailable, falling back to headless mode", file=sys.stderr)

# Load renderer lib if GUI enabled
dll = None
if _HAS_GUI:
    try:
        dll = np.ctypeslib.load_library('render_balls_so', '.')
    except OSError:
        _HAS_GUI = False
        print("show3d_balls: renderer library not found, skipping display", file=sys.stderr)


def showpoints(xyz,
               c_gt=None,
               c_pred=None,
               waittime=0,
               showrot=False,
               magnifyBlue=0,
               freezerot=False,
               background=(0, 0, 0),
               normalizecolor=True,
               ballradius=10):
    """
    Display 3D point clouds using the compiled 'render_balls_so' library.

    On headless or missing dependencies, this is a no-op.
    """
    # Abort early if GUI or renderer not available
    if not _HAS_GUI or dll is None:
        print("show3d_balls.showpoints: GUI/renderer unavailable, skipping display", file=sys.stderr)
        return

    # Center & scale to window
    xyz = xyz - xyz.mean(axis=0)
    radius = np.linalg.norm(xyz, axis=1).max()
    xyz /= (radius * 2.2) / showsz

    # Prepare colour arrays (float32)
    if c_gt is not None:
        c0 = c_gt[:, 0].astype('float32')
        c1 = c_gt[:, 1].astype('float32')
        c2 = c_gt[:, 2].astype('float32')
    else:
        c0 = np.full(len(xyz), 255, dtype='float32')
        c1 = np.full(len(xyz), 255, dtype='float32')
        c2 = np.full(len(xyz), 255, dtype='float32')

    if normalizecolor:
        for arr in (c0, c1, c2):
            maxv = arr.max() + 1e-14
            arr /= (maxv / 255.0)

    # Ensure contiguous
    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        # Rotation matrices
        rot_x = (mousey - 0.5) * np.pi * 1.2 if not freezerot else 0.0
        rot_y = (mousex - 0.5) * np.pi * 1.2 if not freezerot else 0.0

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rot_x), -np.sin(rot_x)],
                       [0, np.sin(rot_x),  np.cos(rot_x)]], dtype='float32')
        Ry = np.array([[ np.cos(rot_y), 0, -np.sin(rot_y)],
                       [0,              1, 0             ],
                       [ np.sin(rot_y), 0,  np.cos(rot_y)]], dtype='float32')
        R = Rx.dot(Ry) * zoom

        # Transform points
        nxyz = xyz.dot(R) + np.array([showsz/2, showsz/2, 0], dtype='float32')
        ixyz = nxyz.astype('int32')

        show[:] = background
        # Call into C renderer
        dll.render_ball(
            ct.c_int(showsz), ct.c_int(showsz),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        # Optional blue magnification
        if magnifyBlue > 0:
            buffs = []
            for shift in ([1,0], [-1,0], [0,1], [0,-1][:magnifyBlue]):
                buffs.append(np.roll(show[:,:,0], shift, axis=(0,1)))
            for b in buffs:
                show[:,:,0] = np.maximum(show[:,:,0], b)

        # Annotate rotation & zoom
        if showrot:
            txt = f"x={int(rot_x/np.pi*180)}°, y={int(rot_y/np.pi*180)}°, z={int(zoom*100)}%"
            cv2.putText(show, txt, (10, showsz-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Main event loop
    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        key = cv2.waitKey(waittime) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        # zoom
        elif key == ord('n'):
            zoom *= 1.1; changed = True
        elif key == ord('m'):
            zoom /= 1.1; changed = True
        elif key == ord('r'):
            zoom = 1.0; changed = True
        # save screen
        elif key == ord('s'):
            cv2.imwrite('show3d.png', show)
        # toggle GT/Pred handled in caller
        if waittime != 0:
            break
    return key


if __name__ == '__main__':
    np.random.seed(100)
    showpoints(np.random.randn(2500,3))