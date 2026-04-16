# Chess Predictor

This project scrapes Swiss tournament data from chess-results.com to build a dataset of player pairings, results, and standings across rounds. An XGBoost model is trained on historical tournament features to predict optimal strategic actions — such as which pairing outcomes to target — for each upcoming Swiss round. SHAP values are used to explain model recommendations and surface the most influential factors per decision.
