# Employee Burnout Prediction

Machine learning project for predicting employee burnout levels (**Low, Medium, High**) based on employee work conditions, productivity, lifestyle, and mental health indicators.

## Dataset

**Remote Work Productivity Dataset 2025** from Kaggle, containing **150,000+ records**.

[Dataset on Kaggle](https://www.kaggle.com/datasets/saitejabandaruin/remote-work-productivity-dataset-2025)

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Approach

* Data preprocessing and exploratory data analysis
* Stratified train-test split
* Feature engineering (`stress_index`, `wellbeing`, categorical features)
* Numerical scaling and categorical encoding
* Model comparison: Decision Tree, SVM, Random Forest, AdaBoost
* Hyperparameter tuning using `GridSearchCV`
* Feature selection using `SelectKBest`
* Feature importance and confusion matrix analysis

## Results

| Model                   |  F1 Macro | Balanced Accuracy |
| ----------------------- | --------: | ----------------: |
| **Best RF (Optimized)** | **0.646** |         **0.652** |
| Random Forest           |     0.641 |             0.647 |
| Decision Tree (Pruned)  |     0.631 |             0.635 |
| AdaBoost                |     0.625 |             0.627 |
| SVM                     |     0.619 |             0.622 |
| Decision Tree (Normal)  |     0.560 |             0.561 |
| Slim RF                 |     0.478 |             0.485 |

The **Optimized Random Forest** achieved the best overall performance, with an **F1-macro of 0.646** and **Balanced Accuracy of 0.652**.

The results show that Random Forest performed best overall, while aggressive feature selection significantly reduced performance.
