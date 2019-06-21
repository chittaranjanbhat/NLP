import argparse
import logging
from logging.config import fileConfig
from src.train import train_pipeline
from config import config

fileConfig(fname='config\\logger.config', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

def main(args):
    if args.train:
        train = train_pipeline.TrainPipeline(config.test_data)
        train.train_preprocess()
    elif args.predict:
        logger.info("Predition started...")
    else:
        logger.warning("Wrong argrument passed, pass -t or -p")


if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--train", "-t", help="To train the model",action='store_true')
    parser.add_argument("--predict", "-p", help="Predict the result using trained model",action='store_true')

    # read arguments from the command line
    args = parser.parse_args()
    main(args)