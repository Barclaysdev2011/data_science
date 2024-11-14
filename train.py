import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Set up argument parsing for SageMaker
    parser = argparse.ArgumentParser()
    
    # Data and model directories
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load training and validation data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    val_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'))
    
    # Separate features and labels
    X_train = train_data.drop(columns=['retained'])
    y_train = train_data['retained']
    X_val = val_data.drop(columns=['retained'])
    y_val = val_data['retained']
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train the model
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(params, dtrain, evals=evals, num_boost_round=100, early_stopping_rounds=10)
    
    # Save the model to the output directory
    model.save_model(os.path.join(args.model_dir, 'xgboost-model'))
