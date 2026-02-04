import argparse
def Get_args():
    parser = argparse.ArgumentParser(description='try_input')
    parser.add_argument('--input1', type=int, help='the first input')
    parser.add_argument('--input2', type=int, help='the second input')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = Get_args()
    print("================================================")
    print(args.input1)
    print(args.input2)
    print(f"add them together and get:{args.input1+args.input2}")
