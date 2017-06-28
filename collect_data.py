import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import grabscreen
w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


file_name = 'outdatanew.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    outdata = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    outdata = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 1]]])

file_name1 = 'slopedatanew.npy'
if os.path.isfile(file_name):
     print('File exists, loading previous data!')
#    slopedata = list(np.load(file_name1))
else:
    print('File does not exist, starting fresh!')
    slopedata = np.array([[[]]])


# vertices = np.array([[0, 480], [10, 300], [80, 225], [560, 225],[630,300],[640,480]], np.int32) 

def regionofinterest(image):
    mask = np.zeros_like(image) # covering a zeros numpy which is a picture with all covered with black
    vertices = np.array([[0, 480], [10, 300], [80, 225], [560, 225],[630,300],[640,480]], np.int32)  # for 640x480 this is the region that we want to process
    #we defined vertices out of the function.
    cv2.fillPoly(mask, [vertices],255) # update mask area with with our vertices, its polygon(yamuk)
    masked = cv2.bitwise_and(image, mask) #with bitwise and, we combine two area one's from our regionofinterest and the other is image
    return masked # to make our image with wanted area, just write regionofinterest(image)


def draw_lines(img,lines):
    if lines is not None:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)

def slope_and_train(lines,slopedata,outdata):
    a0 = ([])
    a2 = ([])
    a3=([])
    a1 = ([])
    
    sums= 0
    if lines is None:
        sums = 0
    if lines is not None:
        for line in lines:
            line 
            m = (line[0][0] - line[0][2]) / (line[0][3] - line[0][1])
            a0.extend([line[0][0]])
            a2.extend([line[0][2]])
            a3.extend([line[0][3]])
            a1.extend([line[0][1]])
        sums = (np.nanmean(a0)-np.nanmean(a2))/((np.nanmean(a3)-np.nanmean(a1))*4000)

    keys = key_check()
    output = np.array(keys_to_output(keys))
    slopedata =  np.append(slopedata,[[[sums]]],axis=0)
    outdata = np.concatenate((outdata,[[output]]),axis=0)
    print("-----pt")
    print(sums)

    return  slopedata,outdata,sums

def just_slope(lines):
    a0 = ([])
    a2 = ([])
    a3=([])
    a1 = ([])
    
    sums= 0
    if lines is None:
        sums = 0
        print("line is none")
    if lines is not None:
        for line in lines:
            
            a0.extend([line[0][0]])
            a2.extend([line[0][2]])
            a3.extend([line[0][3]])
            a1.extend([line[0][1]])
        sums = (np.nanmean(a0)-np.nanmean(a2))/((np.nanmean(a3)-np.nanmean(a1))*4000)
        if sums == np.inf:
            sums = 1
        elif sums ==-np.inf:
            sums=-1
    return sums


def process_img(image):
    lower_yellow = np.array([0, 0, 0])
    upper_yellow = np.array([15, 15, 255])
    # yellow color mask
    processimagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yellow_mask = cv2.inRange(processimagergb, lower_yellow, upper_yellow)  # and we are masking it
    masked = cv2.bitwise_and(processimagergb, processimagergb, mask=yellow_mask)  # and then we combine it with original image
    # turned into gray
    processimagecanny = cv2.Canny(masked, threshold1=0,threshold2=300)  # with canny edge detection method, we detect edges
    # of only our yellow lines' edges. We used masking
    # at the beginning of the code because of this.
    processimagegauss = cv2.GaussianBlur(processimagecanny, (5, 5), 0)  # This'Ll fix some in order to avoid noises
    processedimage = regionofinterest(processimagegauss)  # Let's get back to our predetermined region
    lines = cv2.HoughLinesP(processedimage, 1, np.pi / 180, 5, 100, 70, 15)
    draw_lines(processedimage,lines)

    return processedimage, lines



def process_img_for_conv(image):
    lower_yellow = np.array([0, 0, 0])
    upper_yellow = np.array([15, 15, 255])
    # yellow color mask
    processimagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yellow_mask = cv2.inRange(processimagergb, lower_yellow, upper_yellow)  # and we are masking it
    masked = cv2.bitwise_and(processimagergb, processimagergb, mask=yellow_mask)  # and then we combine it with original image
    # turned into gray
    processimagecanny = cv2.Canny(masked, threshold1=0,threshold2=300)  # with canny edge detection method, we detect edges
    # of only our yellow lines' edges. We used masking
    # at the beginning of the code because of this.
    processimagegauss = cv2.GaussianBlur(processimagecanny, (5, 5), 0)  # This'Ll fix some in order to avoid noises
    processedimage = regionofinterest(processimagegauss)  # Let's get back to our predetermined region

    return processedimage 



def data_for_conv(slopedata,outdata,screen):
  
    keys = key_check()
    output = np.array(keys_to_output(keys))
    slopedata =  np.append(slopedata,[screen])
    outdata = np.concatenate((outdata,[[output]]),axis=0)
    return slopedata,outdata
        







def main(slopedata,outdata):
    for i in list(range(7))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while (True):
        last_time = time.time()
        if not paused:
            # 640*480 windowed mode
            screen = grabscreen.grab_screen(region=(0, 40, 640, 480))

            screen = process_img_for_conv(screen)
            slopedata, outdata, tanhslope = (lines,slopedata,outdata)
           
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        if slopedata.size == 5000:
            print(len(slopedata))
            print("BITTTTIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
            np.save(file_name1, slopedata)
            np.save(file_name, outdata)
        print("--",4999-slopedata.size)    
        

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
                print(tanhslope)
        print('Loop took {} seconds'.format(time.time() - last_time))
        
        
def thatsovter():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
        
    paused = False
    while (True):
        last_time = time.time()
        if not paused:
            # 640*480 windowed mode
            screen = grabscreen.grab_screen(region=(0, 40, 640, 480))

            screen, lines = process_img(screen)
            sums = just_slope(lines)
            
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            print(lines)
            print(sums)
            
            
            





def main_for_conv(slopedata,outdata):
    for i in list(range(7))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while (True):
        last_time = time.time()
        if not paused:
            # 640*480 windowed mode
            screen = grabscreen.grab_screen(region=(0, 40, 640, 480))

            screen = process_img_for_conv(screen)
            screen = cv2.resize(screen,(320,240))
 
            slopedata, outdata = data_for_conv(slopedata,outdata,screen)
           
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
        if slopedata.size == 230400000:
            print(len(slopedata))
            print("BITTTTIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
            np.save(file_name1, slopedata)
            np.save(file_name, outdata)
        print("--",3000-slopedata.size/76800)    
        
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        print('Loop took {} seconds'.format(time.time() - last_time))
     