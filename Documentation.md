# Methodology and Script Documentation

This Documentation outlines the Methodology and Script Documentation for the machine learning workflow to predict the flight behavior of lesser kestrel.It is orientated on the ml_flight_behavior.ipynb notebook and the focus is on the data preperation and preprocessing, Model and Classifaction preparation, model training/evaluation and feature importance.

## 1. Data preparation and Preprocessing

* Outlier Detection
  
  First, we need to determine which data can be used for analysis. This involves visually looking for extreme values and also using the IQR method to identify extreme values. Typically, 3 * IQR is used for strong extreme values, but with a large dataset like the one in this project, many outliers were found even at 4 * IQR.
    ```
    def get_outliers_iqr(df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 4 * iqr
        upper_bound = q3 + 4 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    ```

* Feature Engineering
  
  For good model performance, meaningful features are crucial. In the example of migrant birds, temporal features are important and therefore following values were extracted from timestamp.
  ```
  df_filtered["hour"] = df_filtered["timestamp"].dt.hour
  df_filtered["day"] = df_filtered["timestamp"].dt.day
  df_filtered["month"] = df_filtered["timestamp"].dt.month
  df_filtered["year"] = df_filtered["timestamp"].dt.year
  ```
  Using the kmeans, three geozones (approximately breeding, transit and wintering area) are added to the dataset.
  ```
  kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
  df_filtered['geo_zone'] = kmeans.fit_predict(coords)
  ```

## 2. Model and Classification Preparation

* Classification

  Good classification is crucial, as these labels will be the target variables. Here, a simple differentiation based on ground-speed (km/h) was chosen.
  ```
  def classify_behavior_simple(row):
    if row['ground-speed'] < 3:
        return 0  # resting
    elif 3 <= row['ground-speed'] < 15:
        return 1  # flying
    elif row['ground-speed'] >= 15: 
        return 2  # migrating
  ```

* Dataset Reduction

    Only 10 percent of the original dataset is used for model training (target_frac = 0.1). The dataset is still large enough, so model training doesn't take too long. Additionally, this allows a more balanced dataset with this approach (max(1, target_size // n_classes)).
    ```
    target_frac = 0.1
    target_size = int(len(df_filtered) * target_frac)

    # group by label
    grouped = df_filtered.groupby('label')

    # same number of samples for each label
    n_classes = grouped.ngroups
    samples_per_class = max(1, target_size // n_classes)
    df_balanced_sample = grouped.apply(lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42))
    df_balanced_sample = df_balanced_sample.reset_index(drop=True)
    ```

* splitting training and testing data

    The dataset is split into 80% training and 20% test data. Since we use time-based data, we choose a time-based split to avoid data leakage.

    ```
    split_index = int(0.8 * len(df_sorted))
    split_timestamp = df_sorted.loc[split_index, 'timestamp']

    train_df = df_sorted[df_sorted['timestamp'] <= split_timestamp]
    test_df = df_sorted[df_sorted['timestamp'] > split_timestamp]
    ```

    * Feature Selection

    The dataset is split into categorical and numerical features. The numerical values are scaled for better processing for logistic regression. For the categorical features, the onehotencoder is used so that these features can be processed by the ensemble learners XGBoost and RandomForest
    ```
    categorical = ['month', 'day', 'hour', 'Landcover_Classification', 'geo_zone']
    numerical = ['heading', 'location-long', 'location-lat']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical), # relevant for logisiticregression
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical) # relevant for XGBoost and RandomForest
        ]
    )
    ```

## 3. Model Training and Feature Importance

* Model Training

    Three different models were used for training: two different ensemble learners with RandomForest and XGBoost, and a multinomial logistic regression with an L2 penalty. The logistic regression was used primarily to see how well it performed compared to better-suited models. random_state = 42 is set for reproducibility, and n_jobs = -1 for potentially faster runtime performance.
    ```
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=42, n_jobs=-1) # multinomial and l2 penalty is standard
    }
    ```

    A paramsampler is better suited for large data sets, eliminating the need to implement CV as with RandomizedSearch. CV provides better, more stable results and is especially useful with smaller datasets. This is omitted here to save some runtime, and only 10 iterations with different hyperparameter combinations are selected per model.
    ```
        param_sampler = list(ParameterSampler(
            Hyperparamters[name],
            n_iter=10,
            random_state=42
        ))
    ```
* Model Evaluation

    A classification report and confusion matrix are used for the evaluation. These include the metrics precision, recall, accuracy, and F1 score, which are particularly interesting for the evaluation. Precision and recall provide more detailed information about the individual class predictions. Accuracy and F1 score, on the other hand, better reflect the overall performance and are almost identical, as the class distribution is almost balanced. The F1 macro score was ultimately used to select the best model, which gives each class equal weight in the evaluation, regardless of how unbalanced it is.

* Feature Importance

    Feature importance indicates the importance of individual features, thus revealing even more background information about the model's results. For this purpose, it's important to re-encode the categorical features in Shap, otherwise the actual feature names won't be displayed.
    ```
    onehot_features = best_model.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(categorical)
    ```