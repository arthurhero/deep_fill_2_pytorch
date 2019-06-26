Original idea and implementation in TensorFlow by Jiayu et. al. can be found here: [https://github.com/JiahuiYu/generative_inpainting]

This is an implementation of deep fill v2, not v1. Thus when I tried to implement some of the new features in v2, it might be different from what the authors meant since I implemented them using my own understanding of the paper, as the source code of v2 has not been released yet (at least at the time when I was writing the code).

I added a feature matching loss as some unstabilities appeared during my training. Feel free to delete it.

All the utils and data are not provided. Feel free to create a data folder and utils folder to download your own training data and implement your own utils (mostly just some cv2 calls).


Best,

Ziwen
