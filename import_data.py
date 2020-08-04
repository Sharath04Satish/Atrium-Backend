import wfdb as wf
import numpy as np
from scipy import signal
from biosppy.signals import ecg
from glob import glob

#Gives back a list of all filenames in the directory.
records = glob('data/mitdb/*.atr')

# Get rid of the extension
records = [path[:-4] for path in records]
records.sort()
print('Total files: ', len(records))

realbeats = ['N','L','R','B','A','a','J','S','V','r',
             'F','e','j','n','E','/','f','Q','?', 'P']

for path in records:
    #Split the address of the file
    pathpts = path.split('/')
    #Get the filename only
    fn = pathpts[-1]
    print('Loading file:', path)

    # Read in the data
    # rdsamp - Read WFDB signal files
    record = wf.rdsamp(path)
    #rdann - Read WFDB annotation files to get beat type
    annotation = wf.rdann(path, 'atr')

    # Print some meta informations
    print('Sampling frequency used for this record:', record[1].get('fs'))
    print('Shape of loaded data array:', record[0].shape)
    print('Number of loaded annotations:', len(annotation.num))
    
    # Get the ECG values from the file.
    data = record[0].transpose()

    # Generate the classifications based on the annotations.
    # Get the category of each beat from the annotation
    cat = np.array(annotation.symbol)
    # Create an array of the dimension of cat, to store the numeric value of the beat.
    rate = np.zeros_like(cat, dtype='float')
    
    for catid, catval in enumerate(cat):
        if (catval == 'N'):
            rate[catid] = 1.0 # Normal
        elif (catval == 'L'):
            rate[catid] = 2.0 #LBB
        elif (catval == 'R'):
            rate[catid] = 3.0 #RBB
        elif (catval == 'V'):
            rate[catid] = 4.0 #PVC
        elif (catval == '/'):
            rate[catid] = 5.0 #Paced
            
    rates = np.zeros_like(data[0], dtype='float')
    rates[annotation.sample] = rate
    
    indices = np.arange(data[0].size, dtype='int')

    # Process each channel separately (2 per input file).
    for channelid, channel in enumerate(data):
        chname = record[1].get('sig_name')[channelid]
        print('ECG channel type:', chname)
        
        # Find rpeaks in the ECG data. Most should match with
        # the annotations.
        out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
        rpeaks = np.zeros_like(channel, dtype='float')
        #Wherever a peak is detected, replace 0 with 1.
        rpeaks[out['rpeaks']] = 1.0
        
        beatstoremove = np.array([0])

        # Split into individual heartbeats. 
        beats = np.split(channel, out['rpeaks'])
        for idx, idxval in enumerate(out['rpeaks']):
            firstround = idx == 0
            lastround = idx == len(beats) - 1
            
            # Skip first and last beat.
            if (firstround or lastround):
                continue

            # Get the classification value that is on
            # or near the position of the rpeak index.
            fromidx = 0 if idxval < 10 else idxval - 10
            toidx = idxval + 10
            #Between the given range, find the category that occurs the most.
            catval = rates[fromidx:toidx].max()
            
            # Skip beat if there is no classification.
            if (catval == 0.0):
                beatstoremove = np.append(beatstoremove, idx)
                continue

            catval = catval - 1.0
            if (catval == 4.0):
                    print(path)

            # Append some extra readings from next beat.
            beats[idx] = np.append(beats[idx], beats[idx+1][:40])

            # Normalize the readings to a 0-1 range for ML purposes.
            #ptp - Peak to Peak, max - min
            beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()

            # Resample from 360Hz to 125Hz
            newsize = int((beats[idx].size * 125 / 360) + 0.5)
            beats[idx] = signal.resample(beats[idx], newsize)

            # Skipping records that are too long.
            if (beats[idx].size > 187):
                beatstoremove = np.append(beatstoremove, idx)
                continue

            # Pad with zeroes.
            zerocount = 187 - beats[idx].size
            beats[idx] = np.pad(beats[idx], (0, zerocount), 'constant', constant_values=(0.0, 0.0))

            # Append the classification to the beat data.
            beats[idx] = np.append(beats[idx], catval)

        beatstoremove = np.append(beatstoremove, len(beats)-1)
        
        # Remove first and last beats and the ones without classification.
        beats = np.delete(beats, beatstoremove)

        # Save to CSV file.
        savedata = np.array(list(beats[:]), dtype=np.float)
        outfn = 'data_ecg/' + fn +'_'+ chname + '.csv'
        print('    Generating ', outfn)
        with open(outfn, "wb") as fin:
            np.savetxt(fin, savedata, delimiter=",", fmt='%f')
