import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from cv2 import THRESH_OTSU
from cv2 import watershed

class MathMorphology:
    def __init__(self) -> None:
        pass

    def fill(self,img):#https://github.com/thanhsn/opencv-filling-holes/blob/master/imfill.py
        _, im_th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        a= im_th | im_floodfill_inv
        return a

    def q1(self):
        image = cv2.imread("input/pcb.jpg",0)
        cv2.imshow("00 - Imagem Original", image)

        #  Limiariza a imagem
        _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Realiza um fechamento
        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,struct_element,iterations=2)
        cv2.imshow("01 - fechamento",closing)
        cv2.imwrite("output/q1/01_fechamento.jpg",closing)

        # Preenche a imagem
        filled = (255 * (ndimage.binary_fill_holes(closing))).astype(np.uint8)
        cv2.imshow("02 - preenchimento", filled)
        cv2.imwrite("output/q1/02_preenchimento.jpg",filled)

        diff = filled - closing
        cv2.imshow("03 - Resultado",diff)
        cv2.imwrite("output/q1/3-Resultado.jpg",diff)

        hole,_=cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print("Quantidade de buracos:",len(hole))

        print("Diametro dos buracos:")
        for i in hole: print(np.sqrt(4*cv2.contourArea(i)/np.pi))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q2(self):
        image = cv2.imread("input/morf_test.png",cv2.IMREAD_GRAYSCALE)
        cv2.imshow("00 - Imagem Original", image)

        # destaca o fundo
        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        background = cv2.morphologyEx(image,cv2.MORPH_CLOSE,struct_element)
        cv2.imshow("01 - Fundo",background)
        plt.imsave("output/q2/01_fundo.jpg",background,cmap='gray')

        # remove o fundo
        remove_background = background - image
        cv2.imshow("02 - removendo fundo",remove_background)
        cv2.imwrite("output/q2/02_removefundo.jpg",remove_background)

        # aplicando filtro antes da operação
        bhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,struct_element)
        cv2.imshow("03 - aplicando black hat",bhat)
        cv2.imwrite("output/q2/03_blackhat.jpg",bhat) 


        _,bhat_thresh = cv2.threshold(bhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))

        # aplicando operações morfologicas para melhorar o resultado
        dilation = cv2.dilate(bhat_thresh,struct_element)
        erosion = cv2.erode(dilation,struct_element)
        closing = cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,struct_element)

        result = cv2.fastNlMeansDenoising(closing,None,10,7,40)
        cv2.imshow("04 - resultdo",255 - result)
        cv2.imwrite("output/q2/04_resultado.jpg",255-result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q3(self):
        image_ini = cv2.imread("input/img_cells.jpg")
        image = cv2.cvtColor(image_ini,cv2.COLOR_BGR2GRAY)
        cv2.imshow("00 - Original",image_ini)

        # Binariza a imagem
        _, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
        cv2.imshow("01 - binarizacao",thresh)
        cv2.imwrite("output/q3/01_binarizacao.jpg",thresh)

        # preenche a imagem
        imgFill = self.fill(thresh)
        cv2.imshow("02 - Imagem preenchida",255 - imgFill)
        cv2.imwrite("output/q3/02_preenchimento.jpg",imgFill)


        scruct_element = np.ones((3,3),np.uint8)
        back = cv2.dilate(imgFill,scruct_element,iterations=3)
        # cv2.imshow("bakc",back)

        dist = cv2.distanceTransform(imgFill,cv2.DIST_L2,3)
        cv2.imshow("03 - funcao de distancia",dist)
        cv2.imwrite("output/q3/03_distancia.jpg",dist)

        _,fg = cv2.threshold(dist,0.1*dist.max(),255,0)
        fg = np.uint8(fg)
        # cv2.imshow("fg",fg)

        unk = cv2.subtract(back,fg)
        # cv2.imshow("unk",unk)

        _,markers = cv2.connectedComponents(fg)
        markers +=1
        markers [unk==255]=0
        markers = watershed(image_ini,markers)
        image_ini[markers == -1] = [255,0,0]
        cv2.imshow("04 - resultado",image_ini)
        cv2.imwrite("q3-3-final.jpg",image_ini)

        cv2.waitKey(0)
        cv2.destroyAllWindows()