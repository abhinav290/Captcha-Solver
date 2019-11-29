import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pylab
import librosa.display
import cv2
from pydub import AudioSegment
from PIL import Image, ImageEnhance


def check_dir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


def convert_mp3_to_wav(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.mp3'):
                sound = AudioSegment.from_mp3(os.path.join(os.path.join(root, file)))
                sound.export(os.path.join(root, os.path.splitext(file)[0] + '.wav'), format="wav")


def convert_audio_to_spectrogram(src_dir, out_dir, ext):
    check_dir(out_dir)
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(ext):
                save_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.png')
                if not os.path.exists(save_path):
                    print(root, file)
                    sig, fs = librosa.load(os.path.join(root, file))
                    pylab.figure(figsize=(1.28, 0.96), dpi=100)
                    # Remove axis
                    pylab.axis('off')
                    # Remove the white edge
                    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
                    s = librosa.feature.melspectrogram(y=sig, sr=fs)
                    librosa.display.specshow(librosa.power_to_db(s, ref=np.max))
                    save_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.png')
                    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
                    pylab.close()


def convert_image_to_grayscale(src_path, out_dir, ext):
    if not src_path.endswith(ext):
        return
    img = Image.open(src_path).convert('LA')
    img_contrast = ImageEnhance.Contrast(img)
    save_path = os.path.join(out_dir, os.path.basename(src_path))
    img_contrast.enhance(3).save(save_path)


def convert_images_dir_to_grayscale(src_dir, out_dir, ext):
    check_dir(out_dir)
    if os.path.isdir(src_dir):
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                print(file)
                if not os.path.exists(os.path.join(out_dir, file)):
                    convert_image_to_grayscale(os.path.join(root, file), out_dir, ext)
    else:
        convert_image_to_grayscale(src_dir, out_dir, ext)


def segment_images_dir_to_grayscale(src_dir, out_dir, ext):
    check_dir(out_dir)
    if os.path.isdir(src_dir):
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                print(file)
                if not os.path.exists(os.path.join(out_dir, file)):
                    segment_image(os.path.join(root, file), out_dir, ext)
    else:
        segment_image(src_dir, out_dir, ext)


def segment_image(img_path, out_dir, ext):
    if not img_path.endswith(ext):
        return
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    pylab.figure(figsize=(1.28, 0.64), dpi=100)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.imshow(thresh)
    plt.xticks([]), plt.yticks([])
    save_path = os.path.join(out_dir, os.path.basename(img_path))
    plt.savefig(save_path, bbox_inches=None, pad_inches=0)
    plt.close()


type_data = 'validate'

convert_audio_to_spectrogram(src_dir='./../cian/data/8char-audio-cloud-all-10k',
                             out_dir='./../cian/data/val/specs', ext='mp3')
#convert_images_dir_to_grayscale(src_dir='./../audio_test_data/new_tts/' + type_data + '/specs',
#                                out_dir='./../audio_test_data/new_tts/' + type_data + '/gray', ext='png')


