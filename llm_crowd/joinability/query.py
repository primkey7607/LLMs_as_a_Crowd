import argparse

'''
Depending on the experimental configuration, execute
either join_raw.py or join_experiments.py
'''
def main(experiment):
    raise Exception("Not implemented")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()

    main(args.experiment)
