import argparse


def main(args):





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--dataset", type=str, default="MNIST", help="")
    parser.add_argument("--outdir", type=str, default="./output/", help="")
    parser.add_argument("--epochlen", type=int, default=2, help="")
    args = parser.parse_args()
    main(args)