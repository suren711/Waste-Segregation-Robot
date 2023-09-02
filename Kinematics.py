from math import *

#robot geometry
e = 82.68     #end effector
f = 141.75    #base
re = 310.00   #tricep length
rf = 80.00    #bicep length
 
#trigonometric constants
sqrt3 = sqrt(3.0)
pi = 3.141592653  #PI
sin120 = sqrt3/2.0  
cos120 = -0.5       
tan60 = sqrt3
sin30 = 0.5
tan30 = 1/sqrt3

# function to calculate angle for each arm
def delta_calcAngleYZ(x0,y0,z0): 

    theta = 0.0
    y1 = -0.5 * 0.57735 * f # f/2 * tg 30
    y0 -= 0.5 * 0.57735 * e   # shift center to edge
    # z = a + b*y
    a = (x0*x0 + y0*y0 + z0*z0 +rf*rf - re*re - y1*y1)/(2*z0)
    b = (y1-y0)/z0
    # discriminant
    d = -(a+b*y1)*(a+b*y1)+rf*(b*b*rf+rf)
    #Serial.print(d)
    if (d < 0): 
        print("FAILED(discriminant) ")
        value = [-1.0,theta]
        return value  # non-existing point
     
    yj = (y1 - a*b - sqrt(d))/(b*b + 1) # choosing outer point
    zj = a + b*yj
    if(yj>y1):
        val=180.0
    else: val=0.0
    theta = 180.0*atan(-zj/(y1 - yj))/pi + val
    value = [0.0,theta]
    return value
        
# function to return all the angles according to the x, y and z coordinates
def delta_calcInverse(x0, y0, z0):
 
    angles=[0.0,0.0,0.0]
    VALUE = delta_calcAngleYZ(x0, y0, z0)
    if (VALUE[0] == 0):angles[0]=VALUE[1] 
    if (VALUE[0] == 0): 
        VALUE = delta_calcAngleYZ(x0*cos120 + y0*sin120, y0*cos120-x0*sin120, z0)  # rotate coords to +120 deg
        angles[1] = VALUE[1]
    if (VALUE[0] == 0): 
        VALUE = delta_calcAngleYZ(x0*cos120 - y0*sin120, y0*cos120+x0*sin120, z0)  # rotate coords to -120 deg
        angles[2] = VALUE[1]
    else: angles = [0,0,0]
    return angles



