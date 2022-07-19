import pandas as pd
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

target_columns = ['ZN_PPM']

# design a function to clean each data file and perform feature engineering
def preprocess(data_file, drop_targets):
    logger.info(f"running preprocessing on {data_file}")

    df = pd.read_csv(data_file, index_col=0, parse_dates = True)
    columns = df.columns

    #################
    # insert preprocessing here


    ##################

    try:
        if drop_targets:
            df.drop(columns=target_columns, inplace=True)
    except KeyError:
        pass

    logger.info(f"data after preprocessing has shape of {df.shape}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # add to path to raw data
    parser.add_argument("--input", type=str, default="path/to/raw/data")

    # add path to output data
    parser.add_argument("--output", type = str, default="path/to/out/data")

    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df = preprocess(args.input, True)

    logger.info(f"preprocessed result shape is {df.shape}")

    df.to_csv(args.output)