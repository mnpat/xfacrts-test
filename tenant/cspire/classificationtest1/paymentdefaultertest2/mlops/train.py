import argparse
import logging
import os
import pickle as pkl
import glob
import csv
import traceback

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score



def __read_data(files_path, dataset_percentage=100):
    try:
        logger.info("Reading dataset from source...")

        all_files = glob.glob(os.path.join(files_path, "*.csv"))

        datasets = []

        for filename in all_files:
            data = pd.read_csv(
                filename,
                sep=',',
                header=0
            )

            datasets.append(data)

        data = pd.concat(datasets, axis=0, ignore_index=True)

        data.head()

        data = data.head(int(len(data) * (int(dataset_percentage) / 100)))

        print("Number of records in training data: ",len(data))

        return data
        
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def _xgb_train(X_train,y_train,X_test,y_test,train_hp,is_master,model_dir):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param params: Hyperparameters for XGBoost training.
    :param dtrain: Training dataset.
    :param evals: Evaluation datasets.
    :param num_boost_round: Number of boosting rounds.
    :param model_dir: Directory to save the trained model.
    :param is_master: True if the current node is the master host.
    """
    
    logger.info("Training the Model:")
    
    
    xg_model = XGBClassifier(colsample_bytree= train_hp['colsample_bytree'], gamma= train_hp['gamma'],
                              learning_rate= train_hp['learning_rate'], max_depth= train_hp['max_depth'], n_estimators= train_hp['n_estimators'],
                              reg_alpha= train_hp['reg_alpha'],
                              reg_lambda= train_hp['reg_lambda'], subsample=train_hp['subsample'])
    
    xg_model.fit(X_train, y_train)

    logger.info("Model Trained Successfully.")
    

    if is_master:
        model_location = os.path.join(model_dir, 'xgboost-model')
        pkl.dump(xg_model, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))
        return xg_model

def _xgb_model_perf(xg_model,X_train,X_test,y_train,y_test):
    
    y_train_pred = xg_model.predict(X_train)
    y_test_pred = xg_model.predict(X_test)
    y_pred_proba = xg_model.predict_proba(X_test)
    
    
    print(xg_model)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train, y_train_pred))
    
    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test, y_test_pred))
    
    print('Accuracy')
    print('-------------------------------------------------------')
    print(accuracy_score(y_test, y_test_pred))
    print('')
    
    print('Roc_auc score')
    print('-------------------------------------------------------')
    print(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    print('')
    
    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test, y_test_pred))

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    

    # Hyperparameters are described here
    # parser.add_argument('--max_depth', type=int, default=5)
    # parser.add_argument('--eta', type=float, default=0.2)
    # parser.add_argument('--gamma', type=float, default=4)
    # parser.add_argument('--min_child_weight', type=float, default=6)
    # parser.add_argument('--subsample', type=float, default=0.8)
    # parser.add_argument('--verbosity', type=int, default=0)
    # parser.add_argument('--objective', type=str, default='binary:logistic')
    # parser.add_argument('--num_round', type=int, default=1)


    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--colsample_bytree', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--n_estimators', type=int, default=20)
    parser.add_argument('--reg_alpha', type=float, default=0.2)
    parser.add_argument('--reg_lambda', type=float, default=0.2)
    parser.add_argument('--num_round', type=int, default=1)

    
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args = parser.parse_args()
    logger.info("Arguments: {}".format(args))
    print(f"HP arguments: {args}")



    # dtrain = xgb.DMatrix(train_data.iloc[:, :-1], label=train_data.iloc[:, -1])
    # dval = xgb.DMatrix(val_data.iloc[:, :-1], label=val_data.iloc[:, -1])
    # evals = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]
 
    train_data = __read_data(args.train)
    val_data = __read_data(args.validation)

    target_col = 'target'
    
    y_train = train_data[target_col]
    X_train = train_data.drop(target_col,axis = 1)

    y_test = val_data[target_col]
    X_test = val_data.drop(target_col,axis = 1)

    train_hp = {
        'max_depth': args.max_depth,
        'colsample_bytree': args.colsample_bytree,
        'gamma': args.gamma,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'n_estimators': args.n_estimators,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda
		# 'num_round': args.num_round
    }
    xgb_model = _xgb_train(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_hp=train_hp,
                    is_master=True,
                    model_dir=args.model_dir)
    


    xgb_model_perf = _xgb_model_perf(xgb_model,X_train,X_test,y_train,y_test)

