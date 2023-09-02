from gpiozero import AngularServo
from time import sleep
import pigpio

from gpiozero.pins.pigpio import PiGPIOFactory

# check if the pigpio module is enabled and working
pi = pigpio.pi()
if not pi.connected:
    print("Failed to connect to pigpiod")
    exit()

# create all the servo objects using the pigpio module(for biceps servo) 
factory = PiGPIOFactory()
servo1 = AngularServo(16, min_angle=-90, max_angle=90,min_pulse_width=0.5/1000,max_pulse_width=2.5/1000, pin_factory=factory)
servo2 = AngularServo(20, min_angle=-90, max_angle=90,min_pulse_width=0.5/1000,max_pulse_width=2.5/1000, pin_factory=factory)
servo3 = AngularServo(21, min_angle=-90, max_angle=90,min_pulse_width=0.5/1000,max_pulse_width=2.5/1000, pin_factory=factory)

# the gripper servo
servo0 = AngularServo(19, min_angle=-90, max_angle=90,min_pulse_width=0.5/1000,max_pulse_width=2.5/1000, pin_factory=factory)

#function to set the angle of the servo
def SetAngle(angle1,angle2,angle3):
    global servo1
    global servo2
    global servo3
    
    servo1.angle = angle1
    servo2.angle = angle2
    servo3.angle = angle3
    sleep(0.02)     # ensure proper and smooth function of servo
   
# funstion to release the servos and stop sending a PWM signal
def ReleaseServo():
    global servo1
    global servo2
    global servo3

    global servo0
    
    servo1.detach()
    servo2.detach()
    servo3.detach()

    servo0.detach()

# function to set the position of the grippper
def Gripper(state):

    global servo0

    # set to closed position
    if state == 1:
        servo0.angle = -50
        #sleep(0.02)     # ensure proper and smooth function of servo

    # set to opened positon
    if state == 0:
        servo0.angle = 0
        #sleep(0.02)     # ensure proper and smooth function of servo


#SetAngle(0,0,0)
#Gripper(0)
#ReleaseServo()
    