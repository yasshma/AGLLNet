import os
import numpy as np
import cv2
import math
from glob import glob
from keras.models import load_model
from tqdm import tqdm


def run(input_path, output_path, label_path):
    """Run AGLLNet to enhance the low light image.
    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
    """

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    path = glob(os.path.join(input_path, '*.*'))

    model = load_model('model/AgLLNet.h5')

    for i in tqdm(range(len(path))):

        # data load
        img_A_path = path[i]
        img_A = cv2.imread(img_A_path) / 255.

        if len(img_A.shape) < 3:
            img_A = img_A[:, :, np.newaxis]
            img_A = np.concatenate((img_A, img_A, img_A), axis=2)

        img_A = img_A[:, :, :3]
        W = img_A.shape[0]
        H = img_A.shape[1]
        w = math.ceil(W / (2**7)) * 2**7
        h = math.ceil(H / (2**7)) * 2**7
        img_A = cv2.resize(img_A, (h, w), interpolation=cv2.INTER_LANCZOS4)
        img_A = img_A[np.newaxis, :]

        # forward pass
        out_pred = model.predict(img_A)

        enhance_B = out_pred[0, :, :, 4:7]
        enhance_B = cv2.resize(enhance_B, (H, W), interpolation=cv2.INTER_LANCZOS4)
        enhance_B = np.clip(enhance_B, 0.0, 1.0)

        # output
        filename = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(img_A_path))[0])

        cv2.imwrite(filename + '.png', (enhance_B * 255.).astype(np.uint8))

        label = cv2.imread(label_path)
        result = cv2.imread(filename + '.png')
        psnr = -10*np.log10(np.mean((label-result)**2))
        print(f"\n PSNR metric = {psnr} \n")
        if psnr > 45:
            print("\n TEST PASSED! \n")
        else:
            print("\n TEST NOT PASSED! \n")


if __name__ == "__main__":

    # set paths
    INPUT_PATH = "input"
    OUTPUT_PATH = "output"
    LABEL_PATH = "label.png"

    # compute results
    run(INPUT_PATH, OUTPUT_PATH, LABEL_PATH)
