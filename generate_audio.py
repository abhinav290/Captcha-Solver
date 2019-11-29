from gtts import gTTS
import argparse
import os
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='Length of captchas in characters', type=int, default=8)
    parser.add_argument('--count', help='How many captchas to generate', type=int,default=20000)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str,
                        default='audio_test_data/new/train/mp3')
    parser.add_argument('--symbols', help='File with the symbols to use in captchas',
                        type=str, default='symbols.txt')
    args = parser.parse_args()
    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)


    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    for i in range(args.count):
        captcha_text = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        audio_path = os.path.join(args.output_dir, captcha_text+'.mp3')
        if os.path.exists(audio_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, captcha_text + '_' + str(version) + '.mp3')):
                version += 1
            audio_path = os.path.join(args.output_dir, captcha_text + '_' + str(version) + '.mp3')

        print(i, args.count, captcha_text)
        data = gTTS(captcha_text)
        data.save(audio_path)

if __name__ == '__main__':
    main()