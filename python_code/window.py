from tkinter import *
from photometry import *
from part3 import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from skimage.transform import resize as imresize
from skimage.filters import gaussian
import pickle
import scipy.misc


def btn_clicked1():
    print(entry3.get())
    print(entry2.get())
    print(entry1.get())
    print(entry0.get())
    
    
    #calcul des vecteurs normaux
    img=calcul_needle_map(load_images(entry3.get(), load_intensSources(entry0.get()), load_objMask(entry2.get())), load_lightSources(entry1.get()), load_objMask(entry2.get()), shape=(512, 612, 3))
  
    #ou les lire a partir de fichier .pkl ( necessite au moins une execution)
    #img=read_file("my_img.pkl")
    
    
    img=img.astype(np.uint8)
    cv2.imshow("image vecteurs normals ",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def btn_clicked2():
    img1=read_file("my_img.pkl")  
    img2=cv2.imread("depthmap.png")
    graph3D(img1,"Origin: Normal Map")
    graph3D(img2,"Origin:Depth Map")  


window = Tk()

window.geometry("697x638")
window.configure(bg = "#ecbef0")
window.title("Photometry App")
window.iconbitmap("icon.ico")
canvas = Canvas(
    window,
    bg = "#ecbef0",
    height = 638,
    width = 697,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    215.0, 311.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked1,
    relief = "flat")

b0.place(
    x = 109, y = 463,
    width = 151,
    height = 41)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked2,
    relief = "flat")

b1.place(
    x = 108, y = 531,
    width = 151,
    height = 41)

entry0_img = PhotoImage(file = f"img_textBox0.png")
entry0_bg = canvas.create_image(
    183.5, 148.5,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry0.place(
    x = 51.5, y = 130,
    width = 264.0,
    height = 35)

entry1_img = PhotoImage(file = f"img_textBox1.png")
entry1_bg = canvas.create_image(
    184.5, 244.5,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry1.place(
    x = 52.5, y = 226,
    width = 264.0,
    height = 35)

entry2_img = PhotoImage(file = f"img_textBox2.png")
entry2_bg = canvas.create_image(
    184.5, 337.5,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry2.place(
    x = 52.5, y = 319,
    width = 264.0,
    height = 35)

entry3_img = PhotoImage(file = f"img_textBox3.png")
entry3_bg = canvas.create_image(
    183.5, 427.5,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry3.place(
    x = 51.5, y = 409,
    width = 264.0,
    height = 35)

window.resizable(False, False)
window.mainloop()
