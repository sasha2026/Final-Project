import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def video_data_analysis(time_values, brightness_values, vid_length, video_timespan=[]):


    def extrema(n, arr):
 
        # Empty lists to store points of
        # local maxima and minima
        mx_values = []
        mx_index = []
        mn_values = []
        mn_index = []
    
        # Iterating over all points to check
        # local maxima and local minima
        for i in range(1, n-1):
    
            # Condition for local minima
            if(arr[i-1] > arr[i] < arr[i + 1]):
                mn_index.append(i)
                mn_values.append(arr[i])
            # Condition for local maxima
            elif(arr[i-1] < arr[i] > arr[i + 1]):
                mx_index.append(i)
                mx_values.append(arr[i])
        
        try: # will only run if local minimum and maximums were found

            max_mx = max(mx_values)  
            min_mn = min(mn_values)
            threshold = (max_mx - min_mn) / 4
            # gets a threshold value by subtracting the value of the greatest relative max from the smallest relative min

            count = 0
            index = []
            for i in mx_values:
                try:
                    if i - mn_values[count] > threshold:
                        index.append(count)
                    count += 1
                except IndexError:
                    pass
                # sorts through all maximum to make sure that the distance between 
                # the maximum and the next min is greater than the threshold

                # adds indices of selected maximums to a list

            mx_index = list(np.array(mx_index)[index]) # takes only the indices of the maximum that pass the threshold


            # same thing as above but with minimum
            count2 = 0
            index2 = []
            for i in mn_values:
                try:
                    if mx_values[count2] - i > threshold:
                        index2.append(count2)
                    count2 += 1
                except IndexError:
                    pass

            mn_index = list(np.array(mn_index)[index2])
        
        except ValueError: 
            mx_index.append(np.where(arr == max(arr)))
            mn_index.append(np.where(arr == min(arr)))
            # if program causes a value error because there are no relative max or min
            # the index of the greatest and lowest brightness value are added to the list

        return mx_index, mn_index


    def period(mx, mn):
    
        period_values = []

        # gets period by subtracting the times between each set of consequtive maximum

        for i in range(len(mx)-1):

            period = time_values[mx[i+1]] - time_values[mx[i]]
            # gets period by subtracting the times between each set of consequtive maximum

            period_values.append(period)
            # adds the value for each period to an array of period values
            
        for i in range(len(mx)-1):

            period = time_values[mn[i+1]] - time_values[mn[i]]
            
            period_values.append(period)
            
        average_period = sum(period_values) / len(period_values)
        # finds the average period by adding up the length of each of the periods and dividing by the amount
        
        return average_period


    def data_classifier(brightness_values): # classifies data as DSF or harmonic

        brightness_smooth = signal.savgol_filter(brightness_values, window_length=35, polyorder=5) 
        # smooths the data to eliminate any small variations

        try: # tries to find the extrema of the data set
            mx, mn = extrema(len(brightness_smooth), brightness_smooth)
        except (ValueError, TypeError, IndexError):
            data = "DSF" # if there are no extrema data must be from DSF
        if (len(mx) < 2) and (len(mn) < 2): # if there is only one extrema data is most likely DSF as well
            data = "DSF"
        else:
            data = "harmonic" # if there are 2 or more extrema data is classified as harmonic
        return data



    function = data_classifier(brightness_values) # finds where data is harmonic (from stars) or DSF


    if function == "harmonic":
        
        brightness_smooth = signal.savgol_filter(brightness_values, window_length=35, polyorder=5)
        
        mx, mn = extrema(len(brightness_smooth), brightness_smooth) # gives indices of local extrema
        oscillator_period = period(mx, mn) # finds the oscillator period based on the extrema

        units = "seconds"
        if len(video_timespan) > 0:
            units = video_timespan[1]

        if oscillator_period < 1: # checks to see if the oscillator period is less than one
            if units == "weeks" or units == "week":
                oscillator_period = oscillator_period * 7
                units = "days"
            elif units == "months" or units == "month":
                oscillator_period = oscillator_period * 4
                units = "weeks"
            elif units == "days" or units == "day":
                oscillator_period = oscillator_period * 24
                units = "hours"
            elif units == "hours" or units == "hour":
                oscillator_period = oscillator_period * 60
                units = "minutes"
            elif units == "minutes" or units == "minute":
                oscillator_period = oscillator_period * 60
                units = "seconds"  
                # converts to smaller unit  
                               
        fig, ax1 = plt.subplots()
        ax1.plot(time_values, brightness_smooth, color="red")
        if video_timespan:
            ax1.set_xlabel(video_timespan[1])
        else:
            ax1.set_xlabel("Seconds")
        ax1.set_ylabel("Relative Brightness Units")
        plt.title("Simple Harmonic Oscilator")
        plt.tight_layout()
                
        print("The estimated period of this harmonic oscillator is {} ".format(oscillator_period) + units)
        
        
        
    if function == "DSF":
        
        brightness_smooth = signal.savgol_filter(brightness_values, window_length=35, polyorder=3)
        brightness_rate = signal.savgol_filter(brightness_values, window_length=35, polyorder=2, deriv=1)
        # smooths and takes derivative
            
        fig, ax1 = plt.subplots()
        ax1.plot(time_values, brightness_smooth, color="red", alpha=0.2)
        ax1.set_xlabel("Seconds")
        ax1.set_ylabel("Relative Fluoroscence Units")
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(time_values, brightness_rate, color="green", label = "dF/dT")
        ax2.set_ylabel("dF/dT")
        plt.legend(loc='lower right')
        plt.title("DSF")
        plt.tight_layout()
        
        # there are two methods of getting the denaturation point based on a graph of the fluorescence
        # first method is to find midpoint between max and min and second method is to find inflection point

        #method 1
        mx, mn = extrema(len(brightness_smooth), brightness_smooth) # gives indices
        tm_v1 = time_values[mn[0]] + (time_values[mx[0]] - time_values[mn[0]]) / 2

        #method 2
        mx_deriv, mn_deriv = extrema(len(brightness_rate), brightness_rate)
        tm_v2 = time_values[mx_deriv[0]]
            
        # finds average of both methods
        tm = (tm_v1[0] + tm_v2) / 2
        
        print("This protein denatures at around {} seconds".format(tm))