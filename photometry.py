# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:36:28 2021
@author: chaki
"""

import cv2
import numpy as np 

import pandas as pd
import pickle
import math

#cv2.imread('bbb',cv2.IMREAD_UNCHANGED)
#chaque image devisee par intesite
#convertir au niveau de gris :( B +G + R )/3
#img.reshape

#filenames

file_intesite = "objet1PNG_SII_VISION/light_intensities.txt"
file_direction = "objet1PNG_SII_VISION/light_directions.txt"
file_mask = "objet1PNG_SII_VISION/mask.png"
file_names = "objet1PNG_SII_VISION/filenames.txt"


def load_intensSources(file_intesite):
    
    """retourne un matrice qui represente l intensite de chaque source"""
    
    mat_intesite = pd.read_csv(file_intesite, header = None, sep = " ")
    
    return mat_intesite

def  load_lightSources(file_direction):
    
    """retourne un matrice qui represente la position de chaque source"""
    
    light_sources = pd.read_csv(file_direction,header = None, sep = " ")
    
    return light_sources

def load_objMask(file_mask):
    """retourne une image (numpy array) qui a des 1 sur l'image cible et des 0 sur le fond"""
    
    obj_mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)
    
    if (obj_mask is None):
        print("erreur lors de l ouverture du masque --> return None")
    else:
        #on aura que des  0 et des 1
        obj_mask = obj_mask // 255
    
    #np.array
    return obj_mask

def load_images(filenames, mat_intesite, mask):
    """retourne un matrice qui contient l image dans chaque ligne."""

    #lecture des noms des images
    liste_images = pd.read_csv(filenames, header = None, sep = "\n")
    
    relative_path = "objet1PNG_SII_VISION/"
    
    num_img = 0  
    
    #c'est cette variable qui sera retournee
    obj_images = []  
    
    #on parcourt chaque image
    for name in liste_images.values:
        img = cv2.imread(relative_path+name[0], cv2.IMREAD_UNCHANGED)
        
        #suivre l'avancement 
        print(name)

        #normaliser: changer l interval avec max val (non utilise)
        # max_value = max(img.flatten())
        # img = img / max_value
        
        # -> normaliser avec 2**16-1 (utilise)
        #Dévisser chaque pixel par (2^16 - 1):  changer l’interval des valeurs [0, 2^16 - 1] a [0, 1]
        img = img  / (2**16-1)


        ##
        ####
        ##        
        #diviser chaque pixel sur l intnsite et utiliser la masque pour optimiser puis Convertir l’image en niveau de gris
        #
        #pour parcourir les pixels
        h,w,z = img.shape
        # imgRes: stocker le resultat
        imgRes = np.zeros((img.shape[0],img.shape[1]), np.float32)
        
        #print(name)
        
        for i in range(h):
            for j in range(w):
                
                #utiliser la masque pour optimiser
                #Diviser chaque pixel qui a la valeur 1 dans le masque sur l’intensité de la source
                if(mask[i,j] != 0):
               
                    #intesite blue:0          # R dans la matrice
                    img[i,j,0] = img[i,j,0] / mat_intesite.iloc[num_img,2]
                    #intesite green:1          # G dans la matrice
                    img[i,j,1] = img[i,j,1] / mat_intesite.iloc[num_img,1]
                    #intesite red:2             #B dans la matrice
                    img[i,j,2] = img[i,j,2] / mat_intesite.iloc[num_img,0]
                    

                    #Convertir l’image en niveau de gris
                    imgRes[i,j] = img[i,j,0]* 0.11 +  img[i,j,1]* 0.59 + img[i,j,2]*0.3
                else:
                    imgRes[i,j] = 0
          
        # Redimensionner : 1 image sur une seule ligne
        imgRes = imgRes.reshape(512*612)
        #ajouter a un tableu pour le convertir en numpy.array a l'etape suivante
        obj_images.append(imgRes)
        #pour avoir l'image suivante dans la matrice d'intesite
        num_img += 1    
        
    #transformer la liste en np.array    
    obj_images = np.array(obj_images)
    
    return obj_images
        
        


def changer_interval(normal_pixel):
    """change l interval de [-1,1] a [0,255]"""
    
    #alcul de la norme
    
    norme = math.sqrt(pow(normal_pixel[0],2)+pow(normal_pixel[1],2)+pow(normal_pixel[2],2))
    
    if(norme != 0):
    #    print("norm = ",norme)
        normal_pixel[0] = normal_pixel[0] / norme
        normal_pixel[1] = normal_pixel[1] / norme
        normal_pixel[2] = normal_pixel[2] / norme


    p_inter = np.array(normal_pixel, np.uint8)
    
    
    
      #changer Blue
      
    normal_pixel[0]  =  ((normal_pixel[0] + 1)/2)*255
     
    
    
      #changer Green


    normal_pixel[1]  =  ((normal_pixel[1] + 1)/2)*255
    
    
      #changer Red


    normal_pixel[2]  =  ((normal_pixel[2] + 1)/2)*255
    
    
    ############



    
    return normal_pixel


def calcul_needle_map(obj_images, light_sources, obj_masq, shape=(512, 612, 3)):
    """calcule les vecteurs normaux de chaque pixel et affiche selon x,y,z"""
    
    #pseudo inverse avec SVD
    
    pseudo_inv = np.linalg.pinv(light_sources)
    
    #le nb total de pixel
    nb_pixels = obj_images.shape[1]
    
    #initialiser la matrice des vecteurs normaux
    all_normals = np.zeros(shape, np.float32)

    
    # 1) calcul des normaux
   
    #parcourir tous les pixels de la matrice 'E' (obj_images)

    for j in range(nb_pixels):
        
        pixel = obj_images[:,j]
        
        
        
        #calcule de la position a partir de j, pour inserer dans la matrice des vecteur normaux
        
        ligne = j // shape[1]
        
        col = j % shape[1]
        
        #
        ##
        # produit matriciel 
        #puis
        #changer l interval 
        #
        #utulisation du masque pour optimiser
        #
        # on ne prend que les pixel qui sont dans la zone blanche du masque
        #
        if(obj_masq[ligne, col] != 0):

            normal = np.dot(pseudo_inv, pixel)
            
            normal= changer_interval(normal)
        
            #insertion dans la matrice des vecteurs normaux
            all_normals[ligne, col] = normal
        

       
        
    
    # 2) sauvgarder all_normals sous format .pkl
    save_file("my_img", all_normals)
    
    # changer le type pour afficher
    all_normals = all_normals.astype(np.uint8())
    
    
    # l'affichage se fait a present via l'interface
    #cv2.imshow("image vecteurs normals ",all_normals)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return all_normals
    
    
    
    
def save_file(fname, X):
    """sauvegarde un fichier sous format .pkl pour eviter les calculs a nouveau"""
    with open(fname+".pkl",'wb') as f:
        p = pickle.Pickler(f)
        p.dump(X)
        
def read_file(fname):
    """lit un fichier .pkl"""
    with open(fname,'rb') as f:
        x = pickle.load(f)
    return x
    
    
def main():
    """fonction utilitaire"""
    
    light_sources = load_lightSources(file_direction)
    mat_intesite = load_intensSources(file_intesite)
    mask = load_objMask(file_mask)
    
    obj_images = load_images(file_names, mat_intesite, mask)
    all_normals = calcul_needle_map(obj_images,light_sources,mask)
    
    return light_sources, mat_intesite, mask, obj_images, all_normals
    
