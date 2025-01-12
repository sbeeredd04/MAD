# Market Anomaly Detection (MAD)

## Overview
The Market Anomaly Detection (MAD) project involves building an anomaly detection system that serves as an early warning mechanism for identifying potential financial market crashes before they occur. By utilizing data and machine learning models, the system classifies market conditions and proposes investment strategies for risk mitigation and optimization.

## Current Features
- **Data Loading**: Users can select or upload CSV files as data sources.
- **Feature Engineering**: The application supports raw features, rolling features, and lagging features.
- **Model Training**: Users can train decision trees, random forests, and gradient boosting models.
- **Model Evaluation**: The application provides metrics for model performance, including accuracy, precision, recall, and ROC AUC.
- **Visualization**: Confusion matrices and ROC curves are generated for model evaluation.

## Upcoming Features
- **[DONE]Correlation Filter**: Implement a correlation filter to select relevant features before applying feature engineering.
- **[DONE]Feature Engineering Fixes**: Ensure that all features have the necessary engineered features applied. Apply feature engineering to selected features only and then remove the features that are not needed or used from the multiselected list of features.
- **[DONE]hatbot Integration**: Integrate a chatbot for user interaction and assistance within the application.
- **[DONE]Model Saving**: Implement functionality to save updated models after training.
- **[DONE]Detailed Model Training and Testing**: Review model training and testing processes in detail using Jupyter Notebook files.
- **Auto Tune Feature for Gradient Boosting**: Implementing auto tuning for gradient boosting models.
- **Unsupervised models**: Add unsupervised models and their respective auto tuning parameters along with testing. 
- **Detection Test**: Simulation data and suggesting buy or sell for investement data based on simulated dataset. 

## Issues and Bugs: 
- **Chatbot UI** : fix the chatbot UI and message UI
- **Validation set**: fix the order of validation input to be first and then test set the order is messed up. Thats possibly causing errors. 

## Documentation
### Code Structure
- **`app.py`**: Main application file where the Streamlit interface is defined.
- **`model_functions/`**: Contains various model training functions for decision trees, random forests, gradient boosting, LSTM, and SVM classifiers.
- **`helper.py`**: Utility functions for data loading, feature engineering, and model evaluation.
- **`plotter.py`**: Functions for visualizing model results and performance metrics.

### How to Use
1. **Install Dependencies**: Ensure you have all required libraries installed. You can use `pip install -r requirements.txt`.
2. **Run the Application**: Start the Streamlit application by running `streamlit run app.py` in your terminal.
3. **Load Data**: Use the interface to load your dataset.
4. **Apply Feature Engineering**: Choose the desired feature engineering method.
5. **Train Models**: Select a model and train it on the prepared dataset.
6. **Evaluate Models**: Review the performance metrics and visualizations provided.

## Contribution
Contributions are welcome! Please feel free to submit issues or pull requests for any bugs or enhancements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the contributors and the open-source community for their support and resources.
