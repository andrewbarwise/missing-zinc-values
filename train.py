import argparse
import logging
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join
import pandas as pd
from sklearn.linear_model import LinearRegression
from preprocessing import preprecoss
from lightgbm import LGBMRegressor
from ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train(args):
    logger.info("calling training function")

    df = preprecoss(join(args.data_dir, "public.csv.gz"), False)

    target_columns = ['ZN_PPM']

    y_train = df[target_columns]
    logger.info(f"training target shape is {y_train.shape}")
    X_train = df.drop(columns = target_columns)
    logger.info(f"training input shape is {X_train.shape}")

    # this model predicts the majority class
    lgbm_model = LGBMRegressor()
    lgbm_model.fit(X_train, y_train)

    # quantile loss 
    lower_light = LGBMRegressor(objective='quantile', alpha = 1-0.95)
    lower_light.fit(X_train, y_train)

    upper_light = LGBMRegressor(objective = 'quantile', alpha = 0.95)
    upper_light.fit(X_train, y_train)

    models = [lgbm_model, lower_light, upper_light]

    # save the model to disk
    save_model(EnsembleModel(models), args.model_dir)

def save_model(model, model_dir):