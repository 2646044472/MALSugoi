# MALSugoi

**MALSugoi**: A personalized anime recommendation system using MyAnimeList (MAL) data, powered by collaborative filtering and deep neural networks.

## Project Structure

- **data/**: Raw and processed datasets (anime info and user ratings).
- **experiments/**: Experimental code for testing and evaluating algorithms.
- **recommenders_DNN/**: Deep Neural Network-based recommendation models.
- **recommenders_simple/**: Collaborative filtering-based models.
- **scrapers/**: Scripts to scrape anime and user data from MAL.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Models**:

- **DNN-based Recommendation**: Train the model using `recommenders_DNN/training/training.ipynb` and generate recommendations with `recommenders_DNN/test/test.py`.
- **Collaborative Filtering**: Use `recommenders_simple/collaborative_filtering.ipynb` to train and generate recommendations.