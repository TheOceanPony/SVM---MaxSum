import sys 
import numpy as np
from matplotlib.pyplot import imread, imsave

from utils import compose_set, get_maxd, svm, predict


img_set_sawtooth = {
    "im_l" : np.array( imread("imgs/sawtooth_im2.ppm") ),
    "im_r" : np.array( imread("imgs/sawtooth_im6.ppm") ),
    "im_d" : np.array( imread("imgs/sawtooth_disp2.pgm") )
}

img_set_bull = {
    "im_l" : np.array( imread("imgs/bull_im2.ppm") ),
    "im_r" : np.array( imread("imgs/bull_im6.ppm") ),
    "im_d" : np.array( imread("imgs/bull_disp2.pgm") ),
}

img_set_venus = {
    "im_l" : np.array( imread("imgs/venus_im2.ppm") ),
    "im_r" : np.array( imread("imgs/venus_im6.ppm") ),
    "im_d" : np.array( imread("imgs/venus_disp2.pgm") )
}

img_set_barn = {
    "im_l" : np.array( imread("imgs/barn_im2.ppm") ),
    "im_r" : np.array( imread("imgs/barn_im6.ppm") ),
    "im_d" : np.array( imread("imgs/barn_disp2.pgm") )
}



def main(scale, n_iter):

    # Composing dataset
    training_set   = compose_set([ img_set_barn, img_set_venus, img_set_bull ], scale)
    validation_set = compose_set([ img_set_sawtooth ], scale)
    
    # Max disparity
    maxd = get_maxd(training_set)
    print(f"Dataset is composed.")
    print(f"Info | size: {len(training_set)}, img_shape: {training_set[0][0].shape[:2]}, max_disp: {maxd}")

    # Training
    print("Training...")
    q, g = svm(n_iter, training_set, maxd)

    # Results
    result_img = predict(validation_set[0][0], validation_set[0][1], validation_set[0][2], q, g, flag_l=False)
    print(f"Done! Results written to disk as 'res.png'.")
    imsave('res.png',result_img, cmap='gray')

    return 0


def print_inp_error():
        print("USAGE :python3 maxp.py [scale] [n_iter]")
        print("    [scale]              - scale")
        print("    [n_iter]             - number of iterations")

if __name__ == "__main__":

    # Input check
    if len(sys.argv) < 3:
        print_inp_error()
        sys.exit(1)

    else:
        scale  = int(sys.argv[1])
        n_iter = int(sys.argv[2])
            
    print("")
    print("SVM-MaxSum trainer:")
    print("=================================================")

    main(scale, n_iter)


#img_set_ = {
#    "im_l" : np.array( imread("imgs/") ),
#    "im_r" : np.array( imread("imgs/") ),
#    "im_d" : np.array( imread("imgs/") )
#}