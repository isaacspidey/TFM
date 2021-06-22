# versión de Óscar 21.may/2021
# ajuste de solo una componente: sincrotrón.
#

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
import time


#resolución de mapa
nside = 8 # pixeles de unos 4 grados de tamaño
npix = hp.nside2npix(nside) # numero de pixeles correspondientes al nside de nuestro mapa 


#frecuencias en GHz
#v=np.array([11,13,17,19,33,41,61,94,30,44,70,100,143,217,353]) #ghz
v=np.array([30,44,70,100,143,217,353]) 


C1= np.zeros([npix])
C2= np.zeros([npix])
chi2= np.zeros([npix])
beta = np.zeros([npix])


# Parámetros de modelos:
beta_d= 1.53 #antes teniamos este signo negativo
beta_s= -3

v_0s= 23 #ghz
v_0d= 353 #ghz


T_bb= 2.7e6 #esto está en microkelvin
T_d= 19.2e6 #esto está en microkelvin

#T_s= 100e6 #mapa

# Constantes y valores de conversión:
c= 3e8
h= 6.626e-34 #j s
k_b= 1.38e-29 #kg m2 / s2 microK
factor_v= 1.09
factor_sim = 1.e6

#def B(T_d,v)

#ip1 = np.zeros((7,npix)) # sincrotrón
#ip2 = np.zeros((7,npix)) # polvo
#ip3 = np.zeros((7,npix)) # cmb

ipd = np.zeros((len(v),npix)) # datos
ipe = np.zeros((len(v),npix)) # errores+
ip3 = np.zeros((len(v),npix)) # cmb

path="/home/bearg/simulations/docencia/tfm-ugr2021/"

# Lectura de datos

#CMB
fichero3= path+"pi_cmb_simulation_planck2018_nside8.fits"
ip3_aux = hp.read_map(fichero3,field=0)


#************EL RUIDO ESTA EN MICROKELVIN, EL POLVO EN KELVIN

for inu in range(len(v)):

    #fsyn= path+"pi_syn_beta_cte_freq_"+str(int(v[inu]))+"_nside"+str(int(nside))+".fits"

    #fdust= path+"pi_dust_d0_freq_"+str(int(v[inu]))+"_nside"+str(int(nside))+".fits"
    # LECTURA -SELECCIONAR: (1) SIM SIN RUIDO, (2) SIM. CON RUIDO (3) DATOS
    #fdata= path+"data_simplified_model_freq_"+str(int(v[inu]))+"_nside"+str(int(nside))+".fits"
    fdata= path+"data_simplified_model_freq_"+str(int(v[inu]))+"_nside"+str(int(nside))+"_real_noise.fits"
    #fdata=path+"data/planck_pi_freq_"+str(int(v[inu]))+"_nside"+str(int(nside))+".fits"
    fsigma = path+"noise_data_model_freq_"+str(int(v[inu]))+"_nside"+str(int(nside))+"_real_noise.fits"

    #ip1_aux= hp.read_map(fsyn,field=0)
    #ip2_aux= hp.read_map(fdust,field=0)

    ipd_aux= hp.read_map(fdata,field=0)
    ipe_aux= hp.read_map(fsigma,field=0)

    ipd[inu,:]=ipd_aux[:]
    ipe[inu,:]=ipe_aux[:]


#iteramos para cada pixel
for ipix in range(npix): #Escojo un único pixel
    #La idea sería realizar  la siguiente cuenta para un único pixel elegido dentro de un mapa.
    y = np.linspace(-5., -1., 100) # Partición para el Beta_s
    Z = 0*y # le doy una base de valores nulos al eje z cuyos puntos sustituiremos despues por los Xi^2(beta_d,beta_s)

    sign = 0*y # variable para guardar si los Cs tienen signos negativos.

    C1aux = 0*y
    C2aux = 0*y

    filas=len(Z)

    
    for i in range(filas):
    
        ip1= (v/v_0s)**y[i]  #Modelo de Sincrotron

        ip3= 1.0+ 0.0*v
   

        #RESOLUCIÓN DEL SISTEMA
        #definimos los elementos de matriz
        e11= np.sum((ip3*ip3)/(ipe[:,ipix]*ipe[:,ipix])) 
        e22= np.sum((ip1*ip1)/(ipe[:,ipix]*ipe[:,ipix])) 
        e12=e21= np.sum((ip3*ip1)/(ipe[:,ipix]*ipe[:,ipix])) 

        #este es el término independiente
        e41= np.sum((ip3*ipd[:,ipix])/(ipe[:,ipix]*ipe[:,ipix])) 
        e42= np.sum((ip1*ipd[:,ipix])/(ipe[:,ipix]*ipe[:,ipix]))  

        #Resolvemos el sistema   
        A= np.array([[e11,e12],[e21,e22]])
        B= np.array([e41,e42])
        C= np.linalg.solve(A,B)

        if (C[0] < 0 or C[1] < 0):
            sign[i] = -1

        error=ipd[:,ipix]-(C[0])*ip3-(C[1])*ip1

        Z[i]= np.sum((error*error)/(ipe[:,ipix]*ipe[:,ipix]))  #cálculo del valor de Chi^2 con Beta_1 = X[i][j] y Beta_2 = Y[i][j] empleando el código anterior
        #Z[i]= min(Z[i],1000) # trunco por si queremos visualizar mejor el gráfico
        C1aux[i]= C[0]
        C2aux[i]= C[1]
     
          
#plt.plot(y,Z)
#plt.xlabel("beta_s")
#plt.xlabel("chi^2")
#plt.show()


    min=np.amin(Z)
    #print("ChiMin= ",min)
    chi2[ipix] = min    
    C1opti=C1aux[np.where(Z==np.amin(Z))]
    #print("C1opti =",C1opti)
    C1[ipix] = C1opti
    C2opti=C2aux[np.where(Z==np.amin(Z))]
    #print("C2opti =", C2opti)
    C2[ipix]= C2opti
    beta_smin=y[np.where(Z==np.amin(Z))]
    #print("beta_s= ",beta_smin)
    beta[ipix] = beta_smin


# Visualización
hp.mollview(C1,title="CMB amplitude")
hp.mollview(C2,title="Syn amplitude")
hp.mollview(beta,title="Beta")
hp.mollview(chi2,title="chi2")

plt.show()




#Esto que está comentado abajo es para que me enseñe la superficie con polígonos y de colorines, 

#en vez de mostrar los valores del eje z como lineas de isoalturas



#ax.plot_surface(X, Y, Z, rstride=1, cstride=1,

#                cmap='viridis', edgecolor='none')

#ax.set_title('superficie negra betas')



#fig.show()





#quit()