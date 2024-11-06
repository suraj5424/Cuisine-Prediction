# Recipe Ingredients Dataset Analysis and Cuisine Classification

## Project Overview
This repository contains a detailed analysis of a dataset containing recipes with their respective ingredients, used to predict the cuisine based on these ingredients. The analysis includes various steps such as data exploration, visualizations, and classification modeling using multiple algorithms, including Multinomial Naive Bayes, XGBoost, CNN, and Random Forest.

The primary objective of this project is to explore the relationship between ingredients and cuisine types, and develop machine learning models to classify recipes into different cuisine categories.

---

## Files in the Repository

- **Notebook (`recipe_ingredients_classification.ipynb`)**:
  - This Jupyter notebook contains the entire analysis, including data preprocessing, exploratory data analysis (EDA), feature extraction, and machine learning model training and evaluation.
  - It covers:
    1. **Data Import and Extraction**: Loading and extracting the dataset.
    2. **Data Preprocessing**: Cleaning and organizing data.
    3. **Data Exploration**: Visualizing ingredient distributions and cuisine counts.
    4. **Modeling**: Using multiple classifiers to predict the cuisine of recipes based on ingredients.

- **Data**:
  - The dataset is downloaded from Kaggle using the `kaggle` API and consists of recipes categorized by their cuisine and a list of ingredients.

- **Output**:
  - The notebook generates visualizations, including count plots, word clouds, and confusion matrices, to assess the performance of different models.

---

## Installation and Setup

### Prerequisites
- You need Python 3.x with the following libraries installed:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`
  - `keras`
  - `tensorflow`
  - `wordcloud`
  - `nltk`
  - `numpy`

To install the required libraries, run the following command:
```bash
pip install pandas matplotlib seaborn scikit-learn xgboost keras tensorflow wordcloud nltk numpy
```

### Kaggle API Setup
To download the dataset from Kaggle, you must set up the Kaggle API on your environment.

1. Create a Kaggle account if you don't have one.
2. Go to [Kaggle API](https://www.kaggle.com/docs/api) and create a new API key (a `kaggle.json` file).
3. Upload the `kaggle.json` file to your environment and set the Kaggle credentials path:
   ```python
   !mkdir -p ~/.kaggle
   !cp /content/kaggle.json ~/.kaggle/
   ```

4. Install the Kaggle package:
   ```bash
   pip install kaggle
   ```

5. Download the dataset:
   ```bash
   !kaggle datasets download -d kaggle/recipe-ingredients-dataset
   !unzip -q recipe-ingredients-dataset.zip
   ```

---

## Dataset Information

- **train.json**: The training dataset containing recipes with their ingredients and corresponding cuisine labels.
- **test.json**: The test dataset containing recipes with their ingredients (without cuisine labels for predictions).

Each entry in the dataset consists of:
- `ingredients`: A list of ingredients used in the recipe.
- `cuisine`: The cuisine type for the recipe (only available in the training set).

---

## Key Sections in the Notebook

1. **Data Exploration**:
   - Loading the dataset and performing basic checks.
   - Visualizations:
     - Count plot of cuisines.
     - Distribution of the number of ingredients.
     - Boxplot of ingredients by cuisine.
     - Wordcloud representation of most frequent ingredients per cuisine.

2. **Feature Engineering**:
   - Creating new features like the number of ingredients per recipe.
   - Vectorizing the ingredients list into a bag-of-words representation using `CountVectorizer`.

3. **Machine Learning Models**:
   - **Multinomial Naive Bayes**: A basic model used for classification based on the ingredients.
   - **XGBoost**: A powerful gradient boosting model for classification.
   - **CNN (Convolutional Neural Network)**: A deep learning approach using `Keras` to predict cuisines based on ingredients.
   - **Random Forest**: An ensemble model for predicting cuisines.

   For each model, the notebook includes:
   - Model training
   - Predictions
   - Evaluation metrics (accuracy, confusion matrix, classification report)
   
4. **Model Evaluation**:
   - Confusion matrix visualization for each model.
   - Accuracy comparison across different models.

---

## How to Run the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recipe-ingredients-cuisine-classification.git
   cd recipe-ingredients-cuisine-classification
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook recipe_ingredients_classification.ipynb
   ```

4. Follow the instructions in the notebook to explore the dataset and run the models.

---

## Results and Visualizations

- **Distribution of Cuisines**: Visualizing how the recipes are distributed across different cuisine categories.
- **Number of Ingredients**: Analyzing how the number of ingredients varies across different cuisines.
- **Top 20 Ingredients**: A bar plot showing the most common ingredients in the dataset.
- **Wordclouds**: Word clouds for each cuisine showing the most frequent ingredients.
- **Confusion Matrices**: For each model, a confusion matrix to evaluate the classification performance.

---

## Future Improvements

- **Hyperparameter Tuning**: Use grid search or random search to tune the hyperparameters for each model and improve accuracy.
- **Cross-validation**: Implement cross-validation to ensure the robustness of the models.
- **Deep Learning**: Experiment with more complex deep learning architectures like LSTM or Transformers for ingredient-based classification.
- **Data Augmentation**: Apply data augmentation techniques to expand the dataset and improve model performance.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset sourced from Kaggle: [Recipe Ingredients Dataset](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset)
- Libraries used: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `keras`, `tensorflow`, `wordcloud`, `nltk`, and `numpy`. 

---

## Contact

If you have any questions or suggestions, feel free to open an issue or contact the repository owner at [SV3677781@gmail.com].
