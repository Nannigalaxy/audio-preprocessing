# audio-preprocessing
Audio preprocessing tool for signal processing and machine learning applications.

## Features
- MFCC 
- Audio data split
- Audio augmentation (Random pitch, speed, shift and background overlay)

## For ML dataset loading
### Dataset directory stucture
<pre>
--wav_dataset  
  |--yes  
     |--  y1.wav
     |--  y2.wav
     .
     .
  |--no  
     |--  n1.wav
     |--  n2.wav
    .  
    .  
  |--.background  
     |--  bg1.wav
     |--  bg2.wav
</pre>
     
Category: yes, no, ...  
Background: .background (Need to have same directory name for background as it is hardcoded)

### Example
#### Usage:
```
from audio_preprocess import get_dataset

path = '../input/wav_dataset/'
sampling_rate = 16000
sample_limit = None   # None to use all samples in each category 
seconds = 1           # Audio length in seconds to consider
mfcc_num = 30
mfcc_max_length = 35

X, Y, dataframe = get_dataset(path, sampling_rate, mfcc_num, mfcc_max_length, seconds, sample_limit)
```
