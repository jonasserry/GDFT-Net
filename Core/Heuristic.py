
import scipy.stats as stats
import numpy as np
import time
import IPython
import matplotlib.pyplot as plt

def Heuristic(image,sigma0=10, mem = 5,no_mem = True):
    size = image.shape[0]
    raw_image = image[:,:,0]
    correct_delays = []

    def shift(correct_delays,i,memory):
        if i<mem or no_mem:
            return 1
        else:
            last_delays = correct_delays[i-mem:i]
            sig = np.std(last_delays)+1
            return (np.sqrt(sig))

    previous_delay = np.argmax(raw_image[:,0])
    correct_delays.append(previous_delay)
    i=1
    while i <image.shape[1]:
        col = raw_image[:,i]
        filter =  stats.norm.pdf(np.linspace(int(-size/2),int(size/2),size),loc=previous_delay-size/2 ,scale = sigma0) 
        previous_delay = np.argmax(col*filter)
        correct_delays.append(previous_delay)
        i+=1

    return(np.array(correct_delays)-size/2)


def Heuristic_V2(image,sigma0=10, SN_threshold = 1.4,scaling=np.abs):
    height = image.shape[0]
    length = image.shape[1]
    raw_image = image[:,:,0]
    predicted_delays = []

    temp=[sigma0]

    current_delay = np.argmax(raw_image[:,0])
    predicted_delays.append(current_delay)
    last_good_estimate=-5

    i=1
    while i <length:
        
        
        col = raw_image[:,i]
        broadening_factor = scaling(i-last_good_estimate)*sigma0

                            #potentially smooth how window moves around?
        window =  stats.norm.pdf(np.linspace(int(-height/2),int(height/2),height),loc=current_delay-height/2 ,scale = broadening_factor) 
        current_delay = np.argmax(col*window)
        predicted_delays.append(current_delay)

        SN = col[current_delay]/np.mean(np.delete(col,current_delay))
        if SN>SN_threshold:
            last_good_estimate=i
        
        temp.append(broadening_factor)

        i+=1

        

    return(np.array(predicted_delays)-height/2,temp)



def Hueristic_Images(images,sigma0=10, mem = 5):
    start_time = time.time()
    New_Images = []
    i=0
    out = display(IPython.display.Pretty('Starting'), display_id=True)
    for image in images:
      out.update(IPython.display.Pretty("{0:4.1f}% done".format(i/len(images)*100)))
      New_Images.append(Heuristic(image,sigma0,mem))
      i+=1
    print("Finished | Time taken: %s" % (time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
    return np.array(New_Images)
  
def Hueristic_Images_V2(images,sigma0=10, SN_threshold = 5,scaling = np.abs):
    start_time = time.time()
    New_Images = []
    i=0
    out = display(IPython.display.Pretty('Starting'), display_id=True)
    for image in images:
      out.update(IPython.display.Pretty("{0:4.1f}% done".format(i/len(images)*100)))
      New_Images.append(Heuristic_V2(image,sigma0,SN_threshold,scaling)[0])
      i+=1
    print("Finished | Time taken: %s" % (time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
    return np.array(New_Images)