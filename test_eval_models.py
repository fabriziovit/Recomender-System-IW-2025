import os
import logging
import datetime
from latent_factor_model_recomm import MF_SGD_User_Based
from sklearn.neighbors import NearestNeighbors
from collaborative_filtering_recomm import CollaborativeRecommender
from utils import get_train_valid_test_matrix, load_movielens_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ******************************************************************************************************************** #
def cf_run_eval_mae_rmse():
    # Load the MovieLens dataset
    _, df_ratings, _ = load_movielens_data("dataset/")

    # Create the utility matrix (URM)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    logging.info(f"Utility matrix dimensions: {utility_matrix.shape}")

    # Split into training and test matrix
    train_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, ret_valid=False)

    logging.info(f"Train_matrix dimensions: {train_matrix.shape}")
    logging.info(f"Test_matrix dimensions: {test_matrix.shape}")
    logging.info(f"Training.head() :\n {train_matrix.head()}")
    logging.info(f"Test.head() :\n {test_matrix.head()}")

    # Create directories for results and models
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    evaluation_output: list = []

    NN_list = [600]  # Number of neighbors to consider (could be reduced for testing)
    for NN in NN_list:
        # 5. Initialize the CollaborativeRecommender model **passing the matrix and train_matrix**
        knn_model_pearson_item = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
        knn_model_pearson_user = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
        recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user)  # Pass matrix and train_matrix
        recomm.fit_user_model(train_matrix)

        mae, rmse = recomm.evaluate_mae_rmse(test_matrix)

        evaluation_output.append(f"\### MAE and RMSE Results ###")
        evaluation_output.append(f"  MAE: {mae:.10f}")
        evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

        logging.info(f"\### MAE and RMSE Results ###")
        logging.info(f"  MAE: {mae:.10f}")
        logging.info(f"  RMSE: {rmse:.10f}\n")

        # Save results to file
        model_name = f"new_knn_model_NN{NN}"  # More descriptive model name
        with open(f"results/{model_name}.txt", "w") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
            for line in evaluation_output:
                f.write(line + "\n")
            f.write("=" * 70 + "\n")


# ******************************************************************************************************************** #


def sgd_run_mae_rmse():
    # Load the MovieLens dataset
    _, df_ratings, _ = load_movielens_data("dataset/")

    # Create the utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    logging.info(f"Utility matrix dimensions: {utility_matrix.shape}")

    # Split into training and test matrix
    train_matrix, valid_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, True)

    # Model Parameters
    n_epochs: int = 5000
    num_factors_list: list[int] = [500]
    learning_rate_list: list[float] = [0.001]  # [0.001]
    lambda_list: list[float] = [0.0, 0.001, 0.0001]  # Test with different lambda values (weight decay)

    # Create directories for results and models
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for n_factor in num_factors_list:
        for learning_rate in learning_rate_list:
            for reg in lambda_list:
                evaluation_output = []

                # 5 Matrix Factorization SGD Model
                recomm = MF_SGD_User_Based(n_factor, learning_rate, reg, n_epochs, utility_matrix, train_matrix, valid_matrix)

                # 6. Fit the MF model on Training
                recomm.fit(refit=True, evaluation_output=evaluation_output)

                mae, rmse = recomm.evaluate_mea_rmse(test_matrix)

                evaluation_output.append(f"\### MAE and RMSE Results ###")
                evaluation_output.append(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
                evaluation_output.append(f"  MAE: {mae:.10f}")
                evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

                logging.info(f"\### MAE and RMSE Results ###")
                logging.info(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
                logging.info(f"  MAE: {mae:.10f}")
                logging.info(f"  RMSE: {rmse:.10f}\n")

                # Save results to file
                model_name = f"test_mf_model_n{n_factor}_lr{learning_rate}_lambda{reg}_norm"
                with open(f"results/test/{model_name}.txt", "w") as f:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
                    for line in evaluation_output:
                        f.write(line + "\n")
                    f.write("=" * 70 + "\n")

                # Save each model
                model_path = f"results/test/{model_name}.pkl"
                recomm.save_model(model_path)
                logging.info(f"Model saved to: {model_path}\n")


# ******************************************************************************************************************** #

if __name__ == "__main__":
    # ******************************************
    #! Collaborative Filtering (CF) - Evaluation
    # For RMSE and MAE evaluation
    # cf_run_eval_mae_rmse()
    # For Precision and Recall evaluation
    # cf_run_eval_precision_recall()
    # ******************************************
    #! SGD - Evaluation
    # For RMSE and MAE evaluation
    sgd_run_mae_rmse()
    # For Precision and Recall evaluation
    # sgd_run_precision_recall()
