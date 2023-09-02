from Servo_Control_2 import *
from Kinematics import *

def translate(value, leftmin, leftmax, rightmin, rightmax):
    # figure out how 'wide' each range is
    leftspan = leftmax - leftmin
    rightspan = rightmax - rightmin

    # convert the left range into a 0-1 range (float)
    valuescaled = float(value - leftmin) / float(leftspan)

    # convert the 0-1 range into a value in the right range.
    return rightmin + (valuescaled * rightspan)

def calc_angles(x_value,y_value):
    
    #print the current coordinate selected
    print(f'X={x_value}mm   Y={y_value}mm')
    
    #calculate the angles for the servo given the coordinates
    servo_angles = [0.0, 0.0, 0.0]
    servo_angles=delta_calcInverse(x_value,y_value,-280)

    # return the bicep_angle
    return servo_angles

def move_arm(target_x,target_y,current_x,current_y,increment,delay):
    
    moving = 0
    while current_x != target_x or current_y != target_y:

        # set state of end effector of moving
        moving = 1

        # if the current x and y is not near the target increment the position by a certain value at a time
        if current_x < target_x:
            current_x = min(current_x + increment, target_x)
        else:
            current_x = max(current_x - increment, target_x)
        
        if current_y < target_y:
            current_y = min(current_y + increment, target_y)
        else:
            current_y = max(current_y - increment, target_y)
        
        # set state of end effector of stopped moving
        if current_x == target_x and current_y == target_y: moving = 0
        
        # calculate the angles for the current increment position
        angles=calc_angles(current_x,current_y)
        
        # move end effector to current increment 
        SetAngle(angles[0],angles[1],angles[2])
        # a delay to slow down the end effector speed
        sleep(delay)

        
def pickup(x_new,y_new,x_old,y_old,id1):
    
    # smoothing delay value
    wait = 2

    # var to change magnitude of increment and delay between each increment
    step = 10.0
    delay = 0.1

    # var to set the last position of the end effector
    x_current = 0.0
    y_current = 0.0

    # print the type of item, [0]broccoli, [1]plastic_spoon
    print(f'ID={id1}')

    # translate the center point of the item from camera coordinate to real-life coordinates
    x_target=translate(x_new,0,1,-259,259)
    y_target=translate(y_new,0,1,195,-195) 
    print(f'Target : X={x_target}mm   Y={y_target}mm')
    
    # move the end effector gradually to that point and wait a while
    move_arm(x_target,y_target,x_old,y_old,step,delay)
    sleep(wait)

    # close the gripper and wait a while
    #Gripper(1)
    sleep(wait)

    #set the target drop-off location
    if id1 == 0 :
        x_current = 75
        y_current = 75
    elif id1 == 1 :
        x_current = 75
        y_current = -75

    # move to the drop-off location and wait a while
    move_arm(x_current,y_current,x_target,y_target,step,delay)
    sleep(wait)

    # release the gripper and wait a while
    #Gripper(0)
    sleep(wait)

    # return the last position of the end effector
    return x_current , y_current






