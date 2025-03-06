from data_selection import Conditional_Sampling
from argparse import ArgumentParser

def main(): 
    parser = ArgumentParser()
    parser.add_argument("--path_x", type=str, required=True)
    parser.add_argument("--path_x_prior", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    obj = Conditional_Sampling(path_x=args.path_x, path_x_prior=args.path_x_prior, output_path=args.output_path)
    obj.generate_conditional_dataset(option='mat')

if __name__ == "__main__": 
    main()