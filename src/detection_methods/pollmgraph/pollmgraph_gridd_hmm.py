import logging
import os
import pickle

import numpy as np
from hmmlearn import hmm
from tqdm import tqdm


class PoLLMgraphGriddHMM:

    def __init__(self, dataset: str):
        self.hmm_components_num = 150
        self.hmm_train_iter_num = 40
        self.dataset_name = dataset
        self.model = None

    def train(
        self,
        train_data: list[tuple[list[int], int]],
        validation_data: list[tuple[list[int], int]] = None,
    ):
        train_data = [
            (item["pollm_graph_feature"], item["Hallucinating"]) for item in train_data
        ]
        sequences = [np.array(seq).reshape(-1, 1) for seq, _ in train_data]
        lengths = [len(seq) for seq, _ in train_data]
        X = np.concatenate(sequences, axis=0)

        self.model = hmm.GaussianHMM(
            n_components=self.hmm_components_num,
            n_iter=self.hmm_train_iter_num,
            covariance_type="full",
            init_params="smtc",
            random_state=42,
            verbose=True,
        ).fit(X, lengths)

        # Compute state-to-label mappings for hallucination prediction
        self.compute_state_label_distributions(train_data)

        # self.save_model(
        #     directory="trained_models",
        #     model_name=f"{self.name}_{self.llm_name}_{self.dataset_name}",
        # )

    def predict(self, test_data: list[tuple[list[int], int]], show_report: bool = True):
        true_labels = []
        predicted_labels = []
        predicted_scores = []
        test_data = [
            (item["pollm_graph_feature"], item["Hallucinating"]) for item in test_data
        ]

        for obs_seq, y_true in tqdm(test_data, desc="Evaluating"):
            obs_seq = np.array(obs_seq).reshape(-1, 1)
            states = self.model.predict(obs_seq)

            log_prob_y0 = np.log(self.P_y[0] + 1e-12) + np.sum(
                [np.log(self.P_st_given_y[0][s] + 1e-12) for s in states]
            )
            log_prob_y1 = np.log(self.P_y[1] + 1e-12) + np.sum(
                [np.log(self.P_st_given_y[1][s] + 1e-12) for s in states]
            )

            score = log_prob_y1 - log_prob_y0
            y_pred = 1 if score > 0 else 0

            true_labels.append(y_true)
            predicted_labels.append(y_pred)
            predicted_scores.append(score)

        return predicted_labels, predicted_scores

    def save_model(self, directory: str, model_name: str):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{model_name}.pkl")

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

        logging.info("Saved model")

    def load_model(self):
        path = "pollmgraph/models/.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise NotImplementedError(
                f"Model: {self.name}_{self.llm_name}_{self.dataset_name}"
            )

    def compute_state_label_distributions(
        self, train_data: list[tuple[list[int], int]]
    ):
        """
        Computes P(s_t | y) from training data by running Viterbi decoding on each sequence.
        Stores result in self.P_st_given_y and self.P_y for use in prediction.

        Args:
            train_data: List of (abstract_sequence, label) tuples
        """
        num_states = self.model.n_components
        counts_y = {0: np.zeros(num_states), 1: np.zeros(num_states)}
        total_y = {0: 0, 1: 0}
        label_counts = {0: 0, 1: 0}

        for obs_seq, y in train_data:
            obs_seq = np.array(obs_seq).reshape(-1, 1)
            viterbi_states = self.model.predict(obs_seq)
            for s in viterbi_states:
                counts_y[y][s] += 1
                total_y[y] += 1
            label_counts[y] += 1

        # Normalize to get conditional probabilities
        self.P_st_given_y = {y: counts_y[y] / (total_y[y] + 1e-12) for y in [0, 1]}

        # Prior P(y)
        total_sequences = label_counts[0] + label_counts[1]
        self.P_y = {y: label_counts[y] / total_sequences for y in [0, 1]}

        logging.info("Computed P(s_t | y) and P(y)")
