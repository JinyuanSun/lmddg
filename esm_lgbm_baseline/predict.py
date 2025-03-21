#!/usr/bin/env python
import argparse
import torch
import pandas as pd
import numpy as np
from glob import glob
import lightgbm as lgb
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_embeddings(embedding_dir):
    """Load ESM2 embeddings from the given directory."""
    embedding_dict = {}
    for file in glob(f'{embedding_dir}/*.pt'):
        key = file.split('/')[-1].split('.')[0]
        embedding_dict[key] = torch.load(file)['mean_representations'][33].detach().numpy()
    return embedding_dict

def prepare_data(df, embedding_dict):
    """Prepare data for prediction."""
    X = []
    y = []
    ids = []
    
    for i, row in df.iterrows():
        try:
            wt_key = f'{row["pdb"]}_{row["chain"]}'
            mut_key = f'{row["pdb"]}_{row["chain"]}_{row["mutation"]}'
            
            wt = embedding_dict[wt_key]
            mut = embedding_dict[mut_key]
            
            X.append(wt - mut)
            y.append(row["ddG"])
            ids.append(f"{row['pdb']}_{row['chain']}_{row['mutation']}")
        except KeyError as e:
            logger.warning(f"Skipping {e}: embedding not found")
    
    return np.array(X), np.array(y), ids

def main():
    parser = argparse.ArgumentParser(description="Predict protein stability changes using LightGBM model")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved LightGBM model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the prediction results (CSV)")
    parser.add_argument("--data", type=str, default="../S669.tsv", help="Path to the dataset (default: ../S669.tsv)")
    parser.add_argument("--embedding_dir", type=str, default="esm2_650m_embeds", help="Directory containing embedding files")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Loading model from {args.model}")
    model = lgb.Booster(model_file=args.model)
    
    logger.info(f"Loading dataset from {args.data}")
    df = pd.read_csv(args.data, sep='\t')
    
    logger.info(f"Loading embeddings from {args.embedding_dir}")
    embedding_dict = load_embeddings(args.embedding_dir)
    
    logger.info("Preparing data for prediction")
    X_test, y_test, ids = prepare_data(df, embedding_dict)
    
    logger.info(f"Making predictions on {len(X_test)} samples")
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    pearson_corr, _ = pearsonr(y_test, predictions)
    spearman_corr, _ = spearmanr(y_test, predictions)
    
    # Log evaluation metrics
    logger.info("Test Metrics:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"Pearson correlation: {pearson_corr:.4f}")
    logger.info(f"Spearman correlation: {spearman_corr:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'id': ids,
        'actual_ddG': y_test,
        'predicted_ddG': predictions,
        'error': y_test - predictions
    })
    
    # Save results to CSV
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()