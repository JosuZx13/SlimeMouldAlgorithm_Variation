# -*- coding: utf-8 -*-
"""
PRACTICA TEORIA - METAHEURÍSTICA
Nombre Estudiante: JOSE MANUEL OSUNA LUQUE 20224440-B
"""
import numpy as np
import pandas as pd
import math
from time import time  # Para calcular los tiempos de Ejecucion
import sys
import random
from scipy.spatial import distance  # Permite usar la funcion para calcular la distancia Euclideana

#################################################################
#################################################################
#################################################################

# LECTURA DE DATOS - FUNCIONES AUXILIARES PARA SU COMETIDO
"""
Como la estructura de los archivos es identica y 
todas las filas tienen las mismas columnas, basta con ver cuantas colunmas
tiene una fila, y sacar el total de clases

Se hace una lectura de los ficheros
Leer una linea completa readline()
Para separar se usa split(SEPARADOR, MAXIMO)
Para borrar ese \n al final de un readline, se usará rstrip('\n')
Es necesario pasar los string del vector a numero
vector = list(map(float, vector))
Se hará con un try
"""


# Switch-Case no existe en python pero puede definirse una funcion similar
# Se implementaran unos diccionarios

# Cada posible conjunto de datos queda seleccionado mediante estas funciones
# Así como sus restricciones


# IRIS
def iris_dataset():
    return "iris_set.dat"


def iris_10percent():
    return "iris_set_const_10.const"


def iris_20percent():
    return "iris_set_const_20.const"


#################################################################
# ECOLI
def ecoli_dataset():
    return "ecoli_set.dat"


def ecoli_10percent():
    return "ecoli_set_const_10.const"


def ecoli_20percent():
    return "ecoli_set_const_20.const"


#################################################################
# RAND
def rand_dataset():
    return "rand_set.dat"


def rand_10percent():
    return "rand_set_const_10.const"


def rand_20percent():
    return "rand_set_const_20.const"


#################################################################
# NEWTHYROID
def newthyroid_dataset():
    return "newthyroid_set.dat"


def newthyroid_10percent():
    return "newthyroid_set_const_10.const"


def newthyroid_20percent():
    return "newthyroid_set_const_20.const"


#################################################################

# Son un Switch que selecciona los ficheros de las restricciones

def percent10(tipo):
    return {
        iris_dataset(): iris_10percent(),
        ecoli_dataset(): ecoli_10percent(),
        rand_dataset(): rand_10percent(),
        newthyroid_dataset(): newthyroid_10percent()
    }.get(tipo, error_dataset())


def percent20(tipo):
    return {
        iris_dataset(): iris_20percent(),
        ecoli_dataset(): ecoli_20percent(),
        rand_dataset(): rand_20percent(),
        newthyroid_dataset(): newthyroid_20percent()
    }.get(tipo, error_dataset())


#################################################################

def error_dataset():
    return "Incorrecto"


# Hace uso de las funciones anteriores para automatizar el proceso de eleccion
# O devolver error en caso de no elegir correctamente
# (Como el proceso será automático, no habrá error alguno)
def seleccion_dataset(argumento):
    return {
        1: iris_dataset(),
        2: ecoli_dataset(),
        3: rand_dataset(),
        4: newthyroid_dataset()
    }.get(argumento, error_dataset())


def seleccion_restricciones(argumento, tipo):
    return {
        1: percent10(tipo),
        2: percent20(tipo),
    }.get(argumento, error_dataset())


#################################################################
#################################################################

# LECTURA DE DATOS

# FUNCION DE LECTURA DEL DATASET AUTOMATICA
def LecturaDataset_Automatica(elegir):
    global conjuntoDatos
    global KClusters
    global yLen
    global xLen

    conjuntoDatos = seleccion_dataset(elegir)
    mdatos = []

    try:

        fdatos = open(conjuntoDatos)
        linea = fdatos.readline()
        linea = linea.rstrip('\n').split(',')

        y = list(map(float, linea))

        ncolumnas = len(y)
        yLen = ncolumnas
        xLen = len(fdatos.readlines()) + 1  # porque ya se leyó una linea

        fdatos.seek(0)  # Para devolver el puntero al principio del fichero

        for inicializar in range(xLen):
            mdatos.append([0] * ncolumnas)

        for f in range(xLen):
            linea = fdatos.readline().rstrip('\n').split(',')  # lectura de una linea
            y = list(map(float, linea))  # casting de texto a float
            for c in range(ncolumnas):
                mdatos[f][c] = y[c]

        dfdatos = pd.DataFrame(mdatos)

    finally:
        if elegir == 2:
            KClusters = 8
        else:
            KClusters = 3

        fdatos.close()

    return dfdatos


# FUNCION DE LECTURA DE LAS RESTRICCIONES DEL DATASET ELEGIDO
def LecturaRestricciones_Automatica(conjuntoDatos, elegir):
    global restricciones

    restricciones = seleccion_restricciones(elegir, conjuntoDatos)

    try:
        frestricciones = open(restricciones)
        linea = frestricciones.readline()
        linea = linea.rstrip('\n').split(',')

        y = list(map(int, linea))

        ncolumnas = len(y)
        nfilas = len(frestricciones.readlines()) + 1  # porque ya se leyó una linea

        frestricciones.seek(0)  # Para devolver el puntero al principio del fichero

        mrestricciones = []

        for inicializar in range(nfilas):
            mrestricciones.append([0] * ncolumnas)

        for f in range(nfilas):
            linea = frestricciones.readline().rstrip('\n').split(',')  # lectura de una linea
            y = list(map(int, linea))  # casting de texto a int
            for c in range(ncolumnas):
                mrestricciones[f][c] = y[c]

        dfrestricciones = pd.DataFrame(mrestricciones)

    finally:
        frestricciones.close()

    return dfrestricciones


#################################################################

# VECTORES MUST-LINK & CANNOT-LINK
def VectorizacionML_CL():
    global vectorML
    global vectorCL
    global dataframeRestricciones
    global nRestricciones

    nRestricciones = 0  # Para cuando se encadene en bucle la ejecucion del algoritmo
    # Si Filas = 150, El maximo de restricciones son 11175
    for i in range(xLen):
        for j in range(i + 1, xLen):
            pareja = [i, j]
            res = dataframeRestricciones.iloc[i, j]

            if res == 1:
                vectorML.append(pareja)
                nRestricciones += 1
            elif res == -1:
                vectorCL.append(pareja)
                nRestricciones += 1


# CALCULO DE LA DISTANCIA EUCLIDEANA ENTRE DOS PUNTOS
def DistanciaEuclideana(puntoA, puntoB):
    return distance.euclidean(puntoA, puntoB)


# CALCULO LAMBDA
def CalcularLambda(datas):
    global nRestricciones
    global vLambda

    distMaxima = 0.0
    for i in range(xLen):
        punto1 = datas.iloc[i, 0:yLen].values
        # Haciendo que empiece después de la posicion de la i ahorra comprobaciones innecesarias
        for j in range(i + 1, xLen):
            punto2 = datas.iloc[j, 0:yLen].values

            dist = DistanciaEuclideana(punto1, punto2)

            distMaxima = max(distMaxima, dist)

    # Se quiere el cociente
    vLambda = distMaxima / nRestricciones


# AUXILIAR - COMPRUEBA SI TODOS LOS CLUSTERS TIENEN AL MENOS UN ELEMENTO
def ClustersUnElemento(solucion):
    correcto = True
    for n in range(KClusters):
        cl = n + 1

        # Selecciona los indices de los elementos que cazen TRUE si se cumple la condicion
        auxS = np.transpose(np.nonzero(solucion == cl))

        if len(auxS) == 0:
            correcto = False

    # return correcto
    return correcto


# GENRACION DE LA SOLUCION INICIAL
def GeneracionSolucionInicial():
    # Una solucion del tamaño de la dimension del problema (la matriz de datos)
    # Si tiene 100 filas, la solucion tendrá 100 posiciones
    solucion = np.zeros(xLen, dtype=np.int)
    unElemento = False

    while not unElemento:
        for i in range(xLen):
            # Es un intervalo abierto por la derecha
            # Si hay 3 clusters, debe generar 1, 2, ó 3
            solucion[i] = np.random.randint(1, KClusters + 1)

        unElemento = ClustersUnElemento(solucion)

    return solucion


# Devuelve los indices de un cluster
def ParticionCluster(solucion, k):
    return np.transpose(np.nonzero(solucion == k))


# SE OBTIENEN DOS LISTAS, UNA DONDE SE GUARDAN LOS PUNTOS Y OTRA LOS INDICES, DE CADA CLUSTER
def ObtenerClusters(solucion):
    global dataframeDatos

    # Se hace una copia del dataframe y se asocia la solucion
    df_aux = dataframeDatos.copy()

    df_aux['SOLUCION'] = solucion

    clust = []
    # De paso se guardan los indices de los puntos que pertenecen a un Cluster
    indClust = []
    for k in range(1, KClusters + 1):
        # Toma los indices de los datos que pertenecen al cluster K
        iCl = ParticionCluster(solucion, k)

        mk = df_aux.loc[df_aux['SOLUCION'] == k].iloc[:, 0:yLen].values
        clust.append(mk)

        # Cada fila guarda una lista de indices (La lista 0, el cluster 1)
        indClust.append(iCl)

    return clust, indClust


# CALCULO DEL CENTROIDE DE UN CLUSTER
def CalculoCentroide(cluster):
    dimension = len(cluster)

    cent = np.zeros(yLen)

    for v in cluster:
        cent = cent + v

    if dimension > 1:
        cent = cent / dimension

    return cent


# SE CALCULAN LOS CENTROIDES DEL CROMOSOMA
def ObtenerCentroides(clusters):
    # Se deben calcular los centroides
    cents = np.full((KClusters, yLen), 0.0)

    i = 0
    for ci in clusters:
        cents[i] = CalculoCentroide(ci)
        i += 1

    return cents


# CALCULO DE LAS DISTANCIAS MEDIA INTRA CLUSTER
def DistanciaMediaIntraCluster(clusterI, centroideI):
    # La particion tiene internamente vectores
    """
    Se pide calcular la normal de cada vector del cluster y el centroide suyo
        y elevarlo al cuadrado el resultado. (Diapositvas Seminario 2)
        Despues se divide por el número de elementos del cluster
    calculo de la normal en Python: np.linalg.norm(v - w), por ejemplo
    """
    ndim = len(clusterI)
    sumaNorma = 0.0
    if ndim > 0:

        for ci in clusterI:
            norma = np.linalg.norm(ci - centroideI)
            sumaNorma = sumaNorma + norma

        sumaNorma = math.sqrt(sumaNorma)

        sumaNorma = sumaNorma / ndim

    return sumaNorma


# DESVIACION GENERAL DE LA PARTICION CREADA (Cromosoma)
def DesviacionGeneral(clusters):
    centroides = ObtenerCentroides(clusters)
    desv = 0.0

    # Con zip, elige (si tienen el mismo tamaño) uno y uno en un mismo for
    for ci, mi in zip(clusters, centroides):
        # Devuelve la distancia Media Intra Cluster de un Cluster
        desv = desv + DistanciaMediaIntraCluster(ci, mi)

    # Es la media de las Distancia Intra Cluster
    return desv / KClusters


# HACE UN RECUENTO DE LAS RESTRICCIONES NO CUMPLIDAS MUST-LINK o CANNOT-LINK
def RestML_CL(particiones, vector, tipo):
    restL = 0
    """
    Si la pareja esta en ML y no esta en el cluster, se suma 1
    Si la pareja esta en CL y esta en el cluster, se suma 1
    """
    for pareja in vector:
        booleano = False  # No están juntas

        a = pareja[0]
        b = pareja[1]
        k = 0  # Para poder usar While y controlar que salga cuando el booleano se cumpla

        while k < KClusters and not booleano:
            if a in particiones[k] and b in particiones[k]:
                # Se encuentra en alguna particion juntos
                booleano = True
            # END IF
            k += 1
        # END FOR KLUSTERS

        """
        Si sale del bucle y es el vector MustLink
            booleano == FALSE entonces INCUMPLE, porque NO ESTAN JUNTAS

        Si sale del bucle y es el vector CannotLink
            booleano == TRUE entonces INCUMPLE, porque ESTAN JUNTAS
        """
        if booleano == tipo:
            restL += 1
        # END IF BOOLEANO
    # END FOR VECTORML

    return restL


# INFACTIBILIDAD (RESTRICCIONES INCUMPLIDAS) por SOLUCION
def Infactibilidad(indices):
    # Si el vector es ML, tipo == False --> Incumple si no están juntos
    # Si el vector es CL, tipo == True  --> Incumplen si están juntos
    return RestML_CL(indices, vectorML, False) + RestML_CL(indices, vectorCL, True)


# Funcion Objetivo de una porción del Limo
def FuncionObjetivo(porcion):
    # porcion = una solucion
    clus, ind = ObtenerClusters(porcion)
    desv = DesviacionGeneral(clus)
    infact = Infactibilidad(ind)

    return desv + infact * vLambda, infact, desv


# FUNCION DE RANDOMIZACION DE LAS SEMILLAS
def RandomizadoSemillas():
    sem = []

    valorCorrecto = False

    while not valorCorrecto:

        valorCorrecto = True

        print("Seleecion de Semillas: ")
        print("----------------------------------------------")
        print("1.- Predefinidas ", sem)
        print("2.- Elegidas por el Usuario")
        print("3.- Al Azar (entre 0 y 1000)")
        print("4.- Semilla = 1")
        print("----------------------------------------------")

        opcionSemilla = int(input('Tipo de Elecccion: '))
        # opcionSemilla = 4
        if opcionSemilla == 1:
            sem = [1, 5, 10, 15, 20]
        elif opcionSemilla == 2:
            sem.clear()  # se vacian
            for sm in range(5):
                sK = int(input('Semilla %s: ' % sm))
                sem.append(sK)
        elif opcionSemilla == 3:
            sem = random.sample(range(0, 1000), 5)
        elif opcionSemilla == 4:
            sem.clear()  # se vacian
            sem.append(1)
        else:
            valorCorrecto = False

    return sem


def DatosEjecucion(tiempo, solucion):
    global recogidaDatos

    recogidaDatos.write("\n\t\t\t\tTiempo de Ejecucion: segundos: " + str(tiempo) + " segundos")

    recogidaDatos.write("\n\t\t\t\tDesviacion: %s" % solucion[1])
    recogidaDatos.write("\n\t\t\t\tInfactibilidad: %s" % solucion[2])
    recogidaDatos.write("\n\t\t\t\tObjetivo: %s" % solucion[3])

    recogidaDatos.write("\n")
    recogidaDatos.flush()  # Hace una recogida


#################################################################
#################################################################

# FUNCIONES AUXILIARES PARA EL ALGORITMO

# Actualizar al mejor Fitness, así como el mejor y peor elemento de la iteracion
def ActualizarLimo():
    global slime
    global xBest
    global xBestIter
    global xWorstIter
    global parada

    bI = slime[
        slime.F_OBJETIVO == slime.F_OBJETIVO.min()].index.values  # Elige el menor valor de F_OBJETIVO
    wI = slime[
        slime.F_OBJETIVO == slime.F_OBJETIVO.max()].index.values  # Elige el mayor valor de la columna F_OBJETIVO

    # Es de la Iteracion, por tanto no hay que comparar nada. MIN y MAX ya se encargan de que sean el mejor y peor
    xBestIter = slime.iloc[bI[0]].copy()
    xWorstIter = slime.iloc[wI[0]].copy()

    if xBest.loc['F_OBJETIVO'] > xBestIter.loc['F_OBJETIVO']:
        xBest = xBestIter.copy()

    if xBest.loc['INFACT'] == 0:
        parada = True  # Si en una iteracion no hay cambios, finaliza


# Expansion del Limo --> Se guia por los pesos, por la concentracion de comida
def AproximacionComida():
    global pValue
    global xBestIter
    global slime

    candidato = 0
    mejorV = np.inf

    for w in range(len(slime)):
        fW = w_Values[w] * slime.iloc[w].loc['F_OBJETIVO']

        if fW < mejorV:
            candidato = slime.iloc[w].copy()
            mejorV = np.copy(fW)

    return candidato.loc['PORCION'].copy()


# Formula de Actualización del Peso del Limo para "aprender" ("Saber") qué elemento le beneficia más elegir
def CalcularW():
    global slime
    global upperB
    global xBestIter
    global xWorstIter
    global w_Values

    w_Values = np.zeros(upperB, dtype=np.float64)  # Un clear para recalcular los Pesos

    mitad = int(upperB / 2) + 1
    # xBestIter -> El mejor fitness de la poblacion actual
    # xWorstIter -> El mejor fitness de la poblacion actual
    fBest = xBestIter.loc['F_OBJETIVO']
    inff = (fBest - xWorstIter.loc['F_OBJETIVO'])  # Esta parte no cambia; la columna 3 es la FuncionO

    SmellIndex()  # Se ordena a razón del Fitness

    for fir in range(mitad):
        supp = (fBest - slime.iloc[fir].loc['F_OBJETIVO'])
        logV = (supp / inff) + 1
        rV = RandomValueR()
        w_Values[fir] = 1 - (rV * np.log(logV))

    for sec in range(mitad, upperB):
        supp = (fBest - slime.iloc[sec].loc['F_OBJETIVO'])
        logV = (supp / inff) + 1
        rV = RandomValueR()
        w_Values[sec] = 1 + (rV * np.log(logV))


# Actualizar p
def ActualizarP():
    global pValue
    global slime

    # Se coge el último elemento añadido a la poblacion (Slime)
    sFi = slime.iloc[-1].loc['F_OBJETIVO']

    pValue = np.tanh(AbsolutoTangente(sFi))


# Elegir el elemento del Slime (poblacion) que generará un vecino
def ElegirCandidato():
    global zValue
    global pValue
    global lowerB
    global upperB
    global xBest

    rand = RandomValueR()
    r = RandomValueR()

    if rand < zValue:
        ab = abs(rand * (upperB - lowerB) + lowerB)
        pos = int(ab)
        return slime.iloc[pos].loc['PORCION'].copy()

    elif r < pValue:
        # Elijo uno al azar de toda la poblacion actual guiados por el peso (de los dos elegidos, el mejor)
        return AproximacionComida()

    else:  # r >= pValue:
        # Elijo al mejor de la expansion completa hasta el momento
        return xBest.loc['PORCION'].copy()


# GENERACION DE VECINOS
# Hace un cambio de cluster a un elemento respetando que siga teniendo todos un elemento al menos
def GeneracionVecinos(solucion):
    newV = np.copy(solucion)
    bol = False
    i = np.random.randint(0, len(solucion))
    antiguoValor = np.copy(solucion[i])

    while not bol:
        nuevoValor = np.random.randint(1, KClusters + 1)
        newV[i] = nuevoValor
        bol = (ClustersUnElemento(newV) and antiguoValor != nuevoValor)

    return newV


def Expandir():
    global slime
    global upperB

    x = ElegirCandidato()
    prima = np.copy(GeneracionVecinos(x))
    fPrima, infa, dv = FuncionObjetivo(prima)

    # Se añade a la poblacion
    slime = slime.append({'PORCION': prima, 'DESVIACION': dv, 'INFACT': infa, 'F_OBJETIVO': fPrima}, ignore_index=True)

    ActualizarRangoSlime(False)


# Calcular a para dar valor a Alpha
def CalcularAValue(T):
    arcotan = np.arctanh(-(T / MAX_ITERACIONES)+1)
    return arcotan


# Calcular el valor absoluto del contenido de la tangente
def AbsolutoTangente(sFi):
    global xBest
    fBExpansion = xBest.loc['F_OBJETIVO'].copy()
    return abs(sFi - fBExpansion)


# Genera un valor aleatorio entre 0 y 1
def RandomValueR():
    return random.uniform(0, 1)


# Actualiza el rango del Slime (Modo: True -> reestructura, upper = 1; False upper++)
def ActualizarRangoSlime(modo):
    global upperB
    global lowerB

    if modo:
        upperB = 1
    else:
        upperB += 1
    # Esto siempre se tendrá que hacer, sin importar qué
    lowerB = upperB * (-1)


# Generar una Población Inicial
def CelulaSlime():
    global slime
    global xBest
    global xBestIter
    global xWorstIter

    slime = pd.DataFrame(columns=['PORCION', 'DESVIACION', 'INFACT', 'F_OBJETIVO'])

    porcion = GeneracionSolucionInicial()
    fObjetivo, infa, desvi = FuncionObjetivo(porcion)
    slime = slime.append({'PORCION': porcion, 'DESVIACION': desvi, 'INFACT': infa, 'F_OBJETIVO': fObjetivo},
                         ignore_index=True)

    xBestIter = slime.iloc[0]
    xWorstIter = slime.iloc[0]

    xBest = slime.iloc[0].copy()

    ActualizarRangoSlime(True)


# Ordena la poblacion
def SmellIndex():
    global slime

    # sort_values --> Ordena el DataFrame de menor a mayor; slime.iloc[0] coge el primer elemento
    slime.sort_values(by=['F_OBJETIVO'], inplace=True)


# Capa la Población manteniendo solo la porción del Slime con mejor Fitness
def ReestructurarSlime():
    global slime

    SmellIndex()
    bestPorcion = slime.iloc[0].copy()  # Una vez estan ordenados, el primero es el mejor

    slime = pd.DataFrame(columns=['PORCION', 'DESVIACION', 'INFACT', 'F_OBJETIVO'])

    slime = slime.append(bestPorcion, ignore_index=True)

    ActualizarRangoSlime(True)


def ResetSlime():
    global slime
    global parada
    global xLen
    global yLen
    global KClusters
    global conjuntoDatos
    global restricciones
    global vectorML
    global vectorCL
    global nRestricciones
    global vLambda
    global xBest
    global xBestIter
    global xWorstIter
    global w_Values
    global pValue
    global zValue
    global lowerB
    global upperB

    xBest = 0
    xBestIter = 0
    xWorstIter = 0

    w_Values = np.zeros(0, dtype=np.float64)

    pValue = 0.0
    zValue = 0.03
    lowerB = 0
    upperB = 0

    xLen = 0
    yLen = 0
    KClusters = 0
    conjuntoDatos = ""
    restricciones = ""
    vectorML = []
    vectorCL = []
    nRestricciones = 0
    vLambda = 0.0


##################################################################
#################################################################


# ALGORITMO SLIME MOULD - Diseño Propio
def Diseño_SMA():
    global MAX_ITERACIONES
    global upperB
    global slime
    global parada
    global xBest

    t = 0
    finAlgoritmo = False
    parada = False

    while not finAlgoritmo:

        # Reestructuracion
        if upperB == TAMSLIME:  # La poblacion ha llegado al tamaño máximo
            ReestructurarSlime()

        # Elegir un elemento y generar un vecino
        Expandir()

        ActualizarLimo()

        t += 1

        CalcularW()
        ActualizarP()

        finAlgoritmo = t >= MAX_ITERACIONES or parada

    # FIN DEL WHILE MAX_ITERACIONES
    # RETURN Xbest [Mejor Solución, Desv, Infact y FObjetivo]
    
    return xBest


#################################################################
#################################################################
#################################################################

# VARIABLES GLOBALES DE LOS DATOS

xLen = 0  # Numero de Instancias del conjunto de Datos
yLen = 0  # Numero de Caracteristicas del conjunto de Datos
KClusters = 0  # Tota de Clusters en los que se van a dividir los datos
conjuntoDatos = ""  # Nombre del conjunto de datos leido
restricciones = ""  # Nombre del conjunto de restricciones leido
vectorML = []
vectorCL = []
nRestricciones = 0
vLambda = 0.0

# VARIABLES GLOBALES PARA EL ALGORITMO

MAX_ITERACIONES = 10000
TAMSLIME = 50  # Tamaño de la Población, lo que se expande el limo antes de elegir el camino

slime = pd.DataFrame()  # Población guardaba en un DataFrame

parada = False

xBest = 0  # Localización Individual de la mayor concentración de olor (Que SIEMPRE estará en la expansión)
xBestIter = 0  # Localizacion de la mejor solucion en la iteracion
xWorstIter = 0  # Localizacion de la peor solucion en la iteracion

w_Values = np.zeros(0, dtype=np.float64)  # Peso del Moho de Limo (de cada poblacion existente)

pValue = 0.0
zValue = 0.03
lowerB = 0  # Guarda el tamaño de la población hasta el momento (en valor negativo)
upperB = 0  # Guarda el tamaño de la población hasta el momento (en valor positivo)

#################################################################
#################################################################

# EJECUCIÓN DEL ALGORITMO

# Seleccion de Semillas para el random.seed()
semillas = RandomizadoSemillas()

# Recogida de Datos
recogidaDatos = open("DATOS.dat", "w")
recogidaDatos.write("Semillas Elegidas: %s\n\n" % semillas)


# Inicio de las ejecuciones
for s in semillas:
    print("\tSEMILLA: ", s)
    # Se elige la semilla
    random.seed(s)
    recogidaDatos.write("SEMILLA ACTUAL: %s\n" % s)
    # Hay cuatro conjuntos de datos (range es intervalo semi-cerrado [a, b)
    for iData in range(1, 5):
        print("\t\tLECTURA DE DATOS")
        # Se resetean las variables
        ResetSlime()

        # Se cargan los datos
        dataframeDatos = LecturaDataset_Automatica(iData)
        # Facilita el trato de los datos
        dataframeDatos.index.name = 'index'

        recogidaDatos.write("\n\tDATOS LEIDOS: %s" % conjuntoDatos)
        print("\t\t\tConjunto de Datos: ", conjuntoDatos)

        # Hay dos conjuntos de restricciones, [a, b)
        for iRestr in range(1, 3):
            print("\t\tLECTURA DE RESTRICCIONES")
            dataframeRestricciones = LecturaRestricciones_Automatica(conjuntoDatos, iRestr)

            vectorML = []  # Recogen las posiciones de las instancias que VAN JUNTOS
            vectorCL = []  # Recogen las posiciones de las instancias que VAN SEPARADAS

            VectorizacionML_CL()

            CalcularLambda(dataframeDatos)

            recogidaDatos.write("\n\t\tRESTRICCIONES: %s" % restricciones)
            recogidaDatos.write("\n\t\tLAMBDA: %s" % vLambda)

            print("\t\t\tRestricciones: ", restricciones)

            ###########################################################################################################
            print("\n\t\t\tALGORITMO - SLIME MOULD ALGORITHM (DISEÑO PROPIO)")
            CelulaSlime()
            inicioT = time()
            SMA_Solucion = np.copy(Diseño_SMA())
            ejecucionT = time() - inicioT

            DatosEjecucion(ejecucionT, SMA_Solucion)
            print("\t\t\t\t", ejecucionT)
            print("\t\t\tFIN ALGORITMO - SLIME MOULD ALGORITHM (DISEÑO PROPIO)\n")

            ###########################################################################################################

        # FIN DEL FOR DE LAS RESTRICCIONES POR DATASET
        print("\n\tFIN RESTRICCIONES\n")
        recogidaDatos.write("\n")
    # FIN DEL FOR DEL DATASET
    print("\nFIN DATASET")
    recogidaDatos.write("\n")

# FIN DEL FOR DE LAS SEMILLAS
recogidaDatos.flush()
recogidaDatos.close()

print("\nFINALIZA EL PROCESO")

sys.exit(0)
