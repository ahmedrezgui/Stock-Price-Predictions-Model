from dotenv import load_dotenv
load_dotenv()    

import os
# ensure deploy_model picks up our smoke‐test model
os.environ["MODEL_NAME"] = "test_smoke_model"

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Import your handlers
from legacy.handlers.model_handler  import save_model, get_or_create_model
from legacy.handlers.scaler_handler import save_scaler, get_or_create_scaler

# Names for this smoke-test
TEST_MODEL_NAME  = "test_smoke_model"
TEST_SCALER_NAME = "test_smoke_scaler"

def smoke_test_save_load():
    print("⏳  Registering dummy model & scaler…")
    dummy_model  = LinearRegression()
    dummy_scaler = MinMaxScaler()

    # Save & register in Azure ML
    save_model(dummy_model, TEST_MODEL_NAME)
    save_scaler(dummy_scaler, TEST_SCALER_NAME)

    print("⏳  Loading them back from Azure ML…")
    loaded_model  = get_or_create_model(stock=None, model_name=TEST_MODEL_NAME)
    loaded_scaler = get_or_create_scaler(stock=None, scaler_name=TEST_SCALER_NAME)

    print(f"✅  Loaded types: {type(loaded_model)}, {type(loaded_scaler)}")

def smoke_test_deploy():
    print("⏳  Running deploy_model.py…")
    # This will pick up MODEL_NAME="test_smoke_model" from os.environ
    import scripts.deploy_model  
    print("✅  deploy_model.py finished without error")

if __name__ == "__main__":
    smoke_test_save_load()
    smoke_test_deploy()
    print("🎉  Smoke‐test complete!")
