import plotly.io as pio
pio.renderers.default='browser'
from photometry import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from skimage.transform import resize as imresize
from skimage.filters import gaussian
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PImage


def graph3D(img,txt):

    img=img.astype(np.uint8)

    #change the image to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #modify 02: you can change the last parameter: in this case it's equal to 2
    img=np.sqrt(gaussian(imresize(img,img.shape, order = 2, mode = 'reflect'),1))

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    fig = go.Figure(go.Surface(
        y = yy,
        x = xx,
        z = img))

    fig.update_layout(title=txt)

    fig.show()
