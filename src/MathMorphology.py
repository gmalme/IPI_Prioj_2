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
        cv2.imshow("0 - Imagem Original", image)

        #  Limiariza a imagem
        _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Realiza um fechamento
        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,struct_element,iterations=2)
        cv2.imshow("1 - Imagem depois do fechamento",closing)
        # cv2.imwrite("q1-1-fechamento.jpg",closing)

        # Preenche a imagem
        preenche = (255 * (ndimage.binary_fill_holes(closing))).astype(np.uint8)
        cv2.imshow("2 - Imagem depois do preenchimento", preenche)
        # cv2.imwrite("q1-2-preenchimento.jpg",preenche)

        diff = preenche - closing
        cv2.imshow("3 - Imagem Final",diff)
        # cv2.imwrite("q1-3-furos.jpg",diff)

        furos,_=cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print("Quantidade de furos:",len(furos))

        print("Diametro dos furos:")
        for i in furos: print(np.sqrt(4*cv2.contourArea(i)/np.pi))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q2(self):
        image = cv2.imread("input/morf_test.png",cv2.IMREAD_GRAYSCALE)
        cv2.imshow("0 - Imagem Original", image)

        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        fundo = cv2.morphologyEx(image,cv2.MORPH_CLOSE,struct_element)
        cv2.imshow("1 - Fundo imagem",fundo)
        # cv2.imwrite("q2-1-fundo-cv.jpg",fundo)
        plt.imshow(fundo,cmap='gray')
        plt.show()
        # plt.imsave("q2-1-fundo-plt.jpg",fundo,cmap='gray')

        removeFundo = fundo - image
        cv2.imshow("2 - removendo fundo",removeFundo)
        # cv2.imwrite("q2-2-removefundo.jpg",removeFundo)

        bhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,struct_element)
        cv2.imshow("3 - aplicando black hat",bhat)
        # cv2.imwrite("q2-3-blackhat.jpg",bhat)

        _,bhat_thresh = cv2.threshold(bhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))

        dilatado = cv2.dilate(bhat_thresh,struct_element)
        erosao = cv2.erode(dilatado,struct_element)
        fechamento = cv2.morphologyEx(erosao,cv2.MORPH_CLOSE,struct_element)

        ruido = cv2.fastNlMeansDenoising(fechamento,None,10,7,40)
        cv2.imshow("4 - ruido",255 - ruido)
        # cv2.imwrite("q2-4-ruido.jpg",ruido)
        # cv2.imwrite("q2-4-ruidoinv.jpg",255-ruido)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q3(self):
        image_ini = cv2.imread("input/img_cells.jpg")
        image = cv2.cvtColor(image_ini,cv2.COLOR_BGR2GRAY)
        cv2.imshow("0 - Original",image_ini)

        # realiza limiarização
        _, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+THRESH_OTSU)

        thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
        cv2.imshow("1 - fechamento",thresh)
        # cv2.imwrite("q3-1-fechamento.jpg",thresh)

        imgFill = self.fill(thresh)
        cv2.imshow("2 - Imagem preenchida",255 - imgFill)
        # cv2.imwrite("q3-2-preenchimento.jpg",imgFill)

        scruct_element = np.ones((3,3),np.uint8)

        back = cv2.dilate(imgFill,scruct_element,iterations=3)
        # cv2.imshow("bakc",back)

        dist = cv2.distanceTransform(imgFill,cv2.DIST_L2,3)
        # cv2.imshow("dist",dist)

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
        cv2.imshow("3 - fim",image_ini)
        # cv2.imwrite("q3-3-final.jpg",imageOri)

        cv2.waitKey(0)
        cv2.destroyAllWindows()