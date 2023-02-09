# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"A1": pd.Series([0.0], dtype="float64"), "A2": pd.Series([0.0], dtype="float64"), "A3": pd.Series([0.0], dtype="float64"), "A8": pd.Series([0.0], dtype="float64"), "A9": pd.Series([0.0], dtype="float64"), "A10": pd.Series([0.0], dtype="float64"), "A11": pd.Series([0.0], dtype="float64"), "A12": pd.Series([0.0], dtype="float64"), "A14": pd.Series([0.0], dtype="float64"), "A15": pd.Series([0.0], dtype="float64"), "l": pd.Series([0.0], dtype="float64"), "u": pd.Series([0.0], dtype="float64"), "y": pd.Series([0.0], dtype="float64"), "g": pd.Series([0.0], dtype="float64"), "gg": pd.Series([0.0], dtype="float64"), "p": pd.Series([0.0], dtype="float64"), "aa": pd.Series([0.0], dtype="float64"), "c": pd.Series([0.0], dtype="float64"), "cc": pd.Series([0.0], dtype="float64"), "d": pd.Series([0.0], dtype="float64"), "e": pd.Series([0.0], dtype="float64"), "ff": pd.Series([0.0], dtype="float64"), "i": pd.Series([0.0], dtype="float64"), "j": pd.Series([0.0], dtype="float64"), "k": pd.Series([0.0], dtype="float64"), "m": pd.Series([0.0], dtype="float64"), "q": pd.Series([0.0], dtype="float64"), "r": pd.Series([0.0], dtype="float64"), "w": pd.Series([0.0], dtype="float64"), "x": pd.Series([0.0], dtype="float64"), "bb": pd.Series([0.0], dtype="float64"), "dd": pd.Series([0.0], dtype="float64"), "h": pd.Series([0.0], dtype="float64"), "n": pd.Series([0.0], dtype="float64"), "o": pd.Series([0.0], dtype="float64"), "v": pd.Series([0.0], dtype="float64"), "z": pd.Series([0.0], dtype="float64"), "kk": pd.Series([0.0], dtype="float64"), "mm": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
method_sample = StandardPythonParameterType("predict")

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, pd.DataFrame):
            result = result.values
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
