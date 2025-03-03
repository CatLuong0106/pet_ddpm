import argparse
from data_selection import TestImages

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--test_path", 
                        type=str, 
                        required=True)
    parser.add_argument("--dest_path",
                        type=str,
                        required=True)
    args = parser.parse_args()
    
    obj = TestImages(test_path=args.test_path)
    obj.get_test_images(dest_path=args.dest_path)
    