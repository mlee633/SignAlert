from cmath import pi
import csv
import numpy
import sys

def funct(fileName):
#file = sys.argv[1]
    data = numpy.genfromtxt(fileName, delimiter=',')[1:, :]

    
    return(data)