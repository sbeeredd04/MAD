import os
import json
from datetime import datetime
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chatbot.gemini_bot import init_gemini_chat, generate_gemini_response, chatbot_ui

# Import helper functions from helper.py
from helper import (
    compute_correlations,
    create_time_split,
    get_performance_metrics,
    load_and_analyze_csv,
    get_available_csvs, 
    get_feature_engineering_options,
    apply_feature_engineering,
    save_dataframe_to_csv,
    create_time_split_with_validation,
    train_model
)
from plotter import  plot_confusion_matrix, plot_feature_importances, plot_roc_curve
from model_functions.decision_tree import  auto_tune_decision_tree
from model_functions.random_forest import auto_tune_random_forest
from model_functions.gradient_boosting_classifier import auto_tune_gradient_boosting
import joblib

def main():
    # Set wide layout
    st.set_page_config(layout="wide")

    # Apply custom CSS for padding
    st.markdown(
        """
        <style>
        .main {
            padding: 20px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Adjust column width ratio to 20% for chatbot, 80% for model
    model_col, chatbot_col = st.columns([12, 4])  

    
    with chatbot_col:
        # Display the Gemini chatbot UI
        chatbot_ui()
    
    with model_col:

        #using full width for the title
        st.title("Market Anomaly Detection Model Builder")
        #spacer for the next section for styling
        st.write("\n\n")
        st.write("--------------------------------")

        # 1. Load Data
        st.subheader("1. Load Data")
        data_source = st.radio("Data Source:", ["Select CSV", "Upload CSV"])

        if data_source == "Select CSV":
            # Get available CSV files
            csv_files = get_available_csvs()
            if not csv_files:
                st.warning("No CSV files found in the data directory.")
                st.stop()
            
            # Create selection box with file names
            selected_file = st.selectbox(
                "Select a CSV file:",
                options=[x[0] for x in csv_files],
                format_func=lambda x: x
            )
            
            # Get the full path for the selected file
            selected_path = next(path for name, path in csv_files if name == selected_file)
            
            # Load and analyze the selected file
            df, analysis = load_and_analyze_csv(selected_path)
            
            # Display data summary
            st.write("### Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total Rows: {analysis['total_rows']}")
                st.write(f"Total Columns: {analysis['total_columns']}")
                if analysis['missing_data']:
                    st.write("Columns with missing values:", analysis['missing_data'])
                st.write("Columns with missing values:", analysis['missing_data'])
            
            with col2:
                st.write("Date columns:", analysis['date_columns'])
                st.write("Numeric columns:", len(analysis['numeric_columns']))
                st.write("Categorical columns:", analysis['categorical_columns'])
                st.write("Potential target columns:", analysis['target_columns'])
            
            # Display data preview
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Store the analysis in session state for later use
            st.session_state['data_analysis'] = analysis
            st.session_state['feature_columns'] = analysis['numeric_columns']
            

        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                df, analysis = load_and_analyze_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Store the analysis in session state
                st.session_state['data_analysis'] = analysis
                st.session_state['feature_columns'] = analysis['numeric_columns']
            else:
                st.warning("Please upload a CSV file or choose another option.")
                st.stop()
                
        #save the dataframe to the session state
        st.session_state['df'] = df
        st.session_state['df_original'] = df
                
        
        #spacer for the next section
        st.write("\n\n")
        st.write("--------------------------------")
        
        # 2. Feature Relevance (formerly Feature Engineering)
        st.subheader("2. Feature Relevance")

        # Check if we have data loaded
        if 'df' not in st.session_state:
            
            #show a list of available dataframes in the session state
            #give options to load one of them
            st.write("Available dataframes in session state:")
            st.write(st.session_state.keys())

            st.error("Please load data first.")
            st.stop()

        # Get current dataframe
        df = st.session_state['df']

        # Get numeric columns and potential target columns
        numeric_cols = [col for col in df.columns if df[col].dtype != 'object']
        potential_targets = [col for col in numeric_cols if df[col].nunique() == 2]

        # Select target column
        target_col = st.selectbox(
            "Select Target Column:",
            options=potential_targets,
            help="Select the binary column you want to predict"
        )

        # Update numeric columns to exclude target
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Select correlation method
        corr_method = st.selectbox(
            "Select Correlation Method:",
            options=['pearson', 'spearman', 'kendall'],
            help="""
            - Pearson: Linear correlation
            - Spearman: Monotonic correlation (rank-based)
            - Kendall: Ordinal correlation
            """
        )

        # Add correlation threshold slider
        corr_threshold = st.slider(
            "Correlation threshold (absolute value):", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.05, 
            step=0.01,
            key="correlation_threshold"
        )

        if st.button("Compute Correlations"):
            with st.spinner("Computing correlations..."):
                # Compute correlations
                correlations, corr_dict = compute_correlations(
                    df, 
                    target_col=target_col, 
                    threshold=corr_threshold
                )

                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### All correlations with target")
                    st.dataframe(correlations)
                
                with col2:
                    st.write("### Selected features")
                    st.write(corr_dict['selected_features'])

                # Store in session state
                st.session_state.update({
                    'correlated_features': corr_dict['selected_features'],
                    'correlation_method': corr_method,
                    'target_column': target_col,
                    'correlations': correlations,
                    'selected_features': corr_dict['selected_features']
                })
                
                st.success("Correlation analysis complete!")

        st.write("\n\n")
        st.write("--------------------------------")

        # 3. Feature Engineering (formerly Feature Relevance)
        st.subheader("3. Feature Engineering")

        # Check if correlation analysis has been done
        if 'correlated_features' not in st.session_state:
            st.warning("Please compute correlations first.")
            st.stop()

        # Get feature engineering options
        fe_options = get_feature_engineering_options()
        selected_method = st.selectbox(
            "Select Feature Engineering Method:",
            options=list(fe_options.keys()),
            help="Choose the method to transform your features"
        )

        # Add feature selection for engineering
        selected_fe_cols = st.multiselect(
            "Select features for engineering:",
            options=st.session_state['correlated_features'],
            default=st.session_state['correlated_features'],
            help="Choose which correlated features to transform"
        )

        # Initialize parameters dictionary
        fe_params = {
            'target_column': st.session_state['target_column'],
            'correlation_method': st.session_state['correlation_method'],
            'selected_features': selected_fe_cols  
        }

        # Show method-specific parameters
        if selected_method == 'Rolling Features':
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.number_input(
                    "Rolling Window Size:",
                    min_value=fe_options[selected_method]['params']['window_size']['min'],
                    max_value=fe_options[selected_method]['params']['window_size']['max'],
                    value=fe_options[selected_method]['params']['window_size']['default']
                )
            
            with col2:
                operations = st.multiselect(
                    "Rolling Operations:",
                    options=fe_options[selected_method]['params']['operations']['options'],
                    default=fe_options[selected_method]['params']['operations']['default']
                )
            
            fe_params.update({
                'window_size': window_size,
                'operations': operations
            })

        elif selected_method == 'Lagging Features':
            lag_periods = st.multiselect(
                "Select Lag Periods:",
                options=fe_options[selected_method]['params']['lag_periods']['options'],
                default=fe_options[selected_method]['params']['lag_periods']['default'],
                help="Select the number of periods to lag for each feature"
            )
            
            fe_params.update({
                'lag_periods': lag_periods
            })

        if st.button("Apply Feature Engineering"):
            with st.spinner("Applying feature engineering..."):
                # Apply feature engineering only to correlated features
                transformed_df, feature_summary = apply_feature_engineering(
                    df,
                    selected_fe_cols,  # Use selected features here instead of all correlated features
                    target_col=st.session_state['target_column'],
                    method=selected_method,
                    params=fe_params
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Transformed Data Preview")
                    st.write(f"Shape: {transformed_df.shape}")
                    st.dataframe(transformed_df.head(10))
                
                with col2:
                    st.write("### Feature Summary")
                    st.write(feature_summary['feature_stats'])
                
                # Store results in session state
                st.session_state.update({
                    'transformed_df': transformed_df,
                    'feature_summary': feature_summary,
                    'selected_features': [col for col in transformed_df.columns 
                                        if col not in [st.session_state['target_column'], 'Data']]
                })
                
                st.session_state['feature_engineered_df'] = transformed_df
                
                st.success("Feature engineering applied successfully!")

        st.write("\n\n")
        st.write("--------------------------------")
        
        # 4. Train/Test/Validation Split
        st.subheader("4. Train/Test/Validation Split")
        
        #check if transformed dataframe is in the session state
        if 'transformed_df' not in st.session_state:
            st.error("Please apply feature engineering first before splitting the data.")
            st.stop()
            
        # Get transformed dataframe
        df = st.session_state['transformed_df']
        
        # Convert 'Data' column to datetime if it's not already
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Get dataset date range
        min_date = df['Data'].min()
        max_date = df['Data'].max()
        st.write("Dataset Range:", min_date.strftime('%Y-%m-%d'), "to", max_date.strftime('%Y-%m-%d'))
        
        # Add checkbox for validation split
        include_validation = st.checkbox("Include Validation Split", value=True)
        
        # Initialize or update split percentages when validation toggle changes
        if 'prev_validation_state' not in st.session_state:
            st.session_state.prev_validation_state = include_validation
        
        if st.session_state.prev_validation_state != include_validation:
            if include_validation:
                st.session_state.splits = {
                    'train': 70,
                    'test': 20,
                    'validation': 10
                }
            else:
                st.session_state.splits = {
                    'train': 80,
                    'test': 20
                }
            st.session_state.prev_validation_state = include_validation
        
        # Initialize splits if not exists
        if 'splits' not in st.session_state:
            st.session_state.splits = {
                'train': 70,
                'test': 20,
                'validation': 10
            } if include_validation else {
                'train': 80,
                'test': 20
            }
        
            # Add custom CSS for padding
        st.markdown("""
            <style>
            .stColumn {
                padding: 0 1rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create two columns layout with a spacer column in between
        left_col, spacer, right_col = st.columns([2, 0.2, 1.5])  # Added spacer column with 0.2 relative width

        
        with left_col:
            st.write("### Adjust Split Percentages")
            
            # Train split slider
            train_val = st.slider(
                "Train Split %",
                min_value=40,
                max_value=90,
                value=st.session_state.splits['train'],
                key='train_slider',
                help="Percentage of data to use for training"
            )
            
            # Calculate remaining percentage
            remaining_after_train = 100 - train_val
            
            # Test split slider
            max_test = remaining_after_train if not include_validation else remaining_after_train - 5
            test_val = st.slider(
                "Test Split %",
                min_value=5,
                max_value=max_test,
                value=min(st.session_state.splits['test'], max_test),
                key='test_slider',
                help="Percentage of data to use for testing"
            )
            
            # Validation split slider (if enabled)
            if include_validation:
                remaining_for_val = 100 - train_val - test_val 
                val_val = st.slider(
                    "Validation Split %",
                    min_value=0,
                    max_value=remaining_for_val,
                    value=remaining_for_val,
                    key='val_slider',
                    help="Percentage of data to use for validation"
                )
            
            # Update session state
            st.session_state.splits['train'] = train_val
            st.session_state.splits['test'] = test_val
            if include_validation:
                st.session_state.splits['validation'] = val_val
            elif test_val < remaining_after_train:
                st.session_state.splits['test'] = remaining_after_train
        
        with spacer:
            st.write("")
        
        with right_col:
            st.write("### Split Distribution")
            
            # Create vertical stacked bar chart with smaller size
            fig, ax = plt.subplots(figsize=(0.5, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            # Plot bars vertically
            bottom = 0
            bar_width = 0.15
            
            # Train bar
            ax.bar(0, st.session_state.splits['train'], bottom=bottom, color='#2ecc71',
                width=bar_width, label='Train', capstyle='round')
            bottom += st.session_state.splits['train']
            
            # Test bar
            ax.bar(0, st.session_state.splits['test'], bottom=bottom, color='#e74c3c',
                width=bar_width, label='Test', capstyle='round')
            bottom += st.session_state.splits['test']
            
            # Validation bar (if enabled)
            if include_validation:
                ax.bar(0, st.session_state.splits['validation'], bottom=bottom, color='#3498db',
                    width=bar_width, label='Validation', capstyle='round')
            
            # Customize the plot
            ax.set_ylim(0, 100)
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Add percentage labels on the bars with smaller font
            for container in ax.containers:
                ax.bar_label(container, fmt='%.0f%%', label_type='center', 
                            fontsize=8)
            
            # Add legend below the plot with much smaller font
            ax.legend(bbox_to_anchor=(0.5, -0.05), 
                    loc='upper center', 
                    ncol=8, 
                    fontsize=10,  
                    handlelength=1,
                    handletextpad=0,
                    borderpad=0.2,  
                    labelspacing=0.2)  
            
            plt.tight_layout()
            st.pyplot(fig)

        # Calculate split dates chronologically
        total_days = (max_date - min_date).days
        train_end_date = min_date + pd.Timedelta(days=int(st.session_state.splits['train'] * total_days / 100))

        if include_validation:
            val_end_date = train_end_date + pd.Timedelta(days=int(st.session_state.splits['validation'] * total_days / 100))

        # Create masks chronologically
        train_mask = df['Data'] < train_end_date
        if include_validation:
            val_mask = (df['Data'] >= train_end_date) & (df['Data'] < val_end_date)
            test_mask = df['Data'] >= val_end_date
        else:
            test_mask = df['Data'] >= train_end_date

        # Display sample counts chronologically
        cols = st.columns(3)
        with cols[0]:
            train_samples = len(df[train_mask])
            st.metric("Train Samples", f"{train_samples:,}")

        if include_validation:
            with cols[1]:
                val_samples = len(df[val_mask])
                st.metric("Validation Samples", f"{val_samples:,}")
            
            with cols[2]:
                test_samples = len(df[test_mask])
                st.metric("Test Samples", f"{test_samples:,}")
        else:
            with cols[1]:
                test_samples = len(df[test_mask])
                st.metric("Test Samples", f"{test_samples:,}")

        # Apply data split button
        if st.button("Apply Data Split", help="Click to update the training, validation, and testing sets"):
            with st.spinner("Updating data splits..."):
                selected_features = st.session_state['selected_features']
                
                # Debug prints chronologically
                st.write("### Date Ranges for Splits:")
                st.write(f"Training dates: {df[train_mask]['Data'].min()} to {df[train_mask]['Data'].max()}")
                if include_validation:
                    st.write(f"Validation dates: {df[val_mask]['Data'].min()} to {df[val_mask]['Data'].max()}")
                    st.write(f"Testing dates: {df[test_mask]['Data'].min()} to {df[test_mask]['Data'].max()}")
                else:
                    st.write(f"Testing dates: {df[test_mask]['Data'].min()} to {df[test_mask]['Data'].max()}")
                
                # Store split data
                if include_validation:
                    X_train, X_val, X_test, y_train, y_val, y_test = create_time_split_with_validation(
                        df, 
                        target_col=target_col,
                        selected_features=selected_features,
                        train_end_date=train_end_date,
                        val_end_date=val_end_date
                    )
                    
                    st.write("### Data Shapes:")
                    st.write(f"X_train shape: {X_train.shape}")
                    st.write(f"X_val shape: {X_val.shape}")
                    st.write(f"X_test shape: {X_test.shape}")
                    
                    st.session_state.update({
                        'X_train': X_train,
                        'X_val': X_val,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_val': y_val,
                        'y_test': y_test,
                        'include_validation': include_validation,
                        'data_split_applied': True
                    })
                else:
                    X_train, X_test, y_train, y_test = create_time_split(
                        df,
                        target_col=target_col,
                        selected_features=selected_features,
                        split_date=train_end_date
                    )
                    
                    st.write("### Data Shapes:")
                    st.write(f"X_train shape: {X_train.shape}")
                    st.write(f"X_test shape: {X_test.shape}")
                    
                    st.session_state.update({
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'include_validation': include_validation,
                        'data_split_applied': True
                    })
                st.success("Data split updated successfully!")


        
        
        #spacer for the next section
        st.write("\n\n")
        st.write("--------------------------------")
        
        # 5. Modeling
        st.subheader("5. Model Training & Evaluation")

        # Create tabs for model info and training
        training_tab, model_tab = st.tabs(["Model Training", "Model(s) Information"])

        with training_tab:
            # Model selection and training section
            st.markdown("### Model Configuration")

            # Add unique keys to all sliders
            model_type = st.selectbox(
                "Model Type", 
                [
                    "decision_tree",
                    "random_forest", 
                    "gradient_boost", 
                    "logistic_regression",
                    "lstm (BETA) ⚠️",  # Added BETA label and warning emoji
                    "svm (BETA) ⚠️",
                    "tcn (BETA) ⚠️"
                ],
                key="model_type_select",
                help="Models marked with (BETA) ⚠️ are experimental and may contain bugs"
            )
            
            # Clean the model type string if needed for processing
            model_type = model_type.split(" ")[0] if "(BETA)" in model_type else model_type

            # Show warning if beta model is selected
            if "(BETA)" in model_type:
                st.warning("⚠️ You've selected a model that's currently in beta. It may be unstable or contain bugs.")

            # Add logistic regression specific parameters
            if model_type == "logistic_regression":
                C = st.slider(
                    "Inverse of regularization strength (C)", 
                    0.01, 10.0, 1.0, 0.01,
                    key="lr_c"
                )
                solver = st.selectbox(
                    "Solver",
                    ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                    key="lr_solver"
                )
                max_iter = st.slider(
                    "Maximum iterations",
                    100, 1000, 100, 50,
                    key="lr_max_iter"
                )
                model_params = {
                    'C': C,
                    'solver': solver,
                    'max_iter': max_iter
                }

            if model_type == "decision_tree":
                
                # Add auto-tune checkbox
                use_auto_tune = st.checkbox("Use Auto-Tuning", 
                    help="Automatically try different combinations of parameters to find the best model")
                
                if not use_auto_tune:
                    # Original manual parameter selection
                    max_depth_dt = st.slider("max_depth", 1, 20, 5, 1, key="dt_depth")
                    min_samples_split_dt = st.slider("min_samples_split", 2, 20, 5, 1, key="dt_split")
                    model_params = {
                        'max_depth': max_depth_dt,
                        'min_samples_split': min_samples_split_dt
                    }
                else:
                    st.info("Auto-tuning will try multiple parameter combinations to find the best model. "
                        "This may take a few minutes.")
                    model_params = {'auto_tune': True}
                
            elif model_type == "random_forest":
                # Add auto-tune checkbox
                use_auto_tune_rf = st.checkbox("Use Auto-Tuning", 
                    help="Automatically try different combinations of parameters to find the best model")
                
                if not use_auto_tune_rf:
                    # Original manual parameter selection
                    n_estimators_rf = st.slider("n_estimators", 10, 300, 100, 10, key="rf_estimators")
                    max_depth_rf = st.slider("max_depth", 1, 20, 10, 1, key="rf_depth")
                    min_samples_split_rf = st.slider("min_samples_split", 2, 20, 5, 1, key="rf_split")
                    model_params = {
                        'n_estimators': n_estimators_rf,
                        'max_depth': max_depth_rf,
                        'min_samples_split': min_samples_split_rf
                    }
                else:
                    st.info("Auto-tuning will try multiple parameter combinations to find the best model. "
                            "This may take a few minutes.")
                    model_params = {'auto_tune': True}

            elif model_type == "gradient_boost":

                # Select the framework
                framework = st.selectbox(
                    "Gradient Boosting Framework",
                    ["sklearn", "xgboost", "lightgbm", "catboost"],
                    help="""
                    - sklearn: Scikit-learn's GradientBoostingClassifier
                    - xgboost: XGBoost (faster, more features)
                    - lightgbm: LightGBM (faster, handles large datasets better)
                    - catboost: CatBoost (handles categorical variables automatically)
                    """
                )

                # Add auto-tune checkbox
                use_auto_tune_gb = st.checkbox(
                    "Use Auto-Tuning",
                    help="Automatically try different combinations of parameters to find the best model"
                )

                if not use_auto_tune_gb:
                    # ==========
                    # Manual Parameter Selection
                    # ==========

                    n_estimators = st.slider(
                        "Number of estimators",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        key="gb_n_estimators"
                    )

                    learning_rate = st.slider(
                        "Learning rate",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.1,
                        step=0.001,
                        format="%.3f",
                        key="gb_learning_rate"
                    )

                    max_depth = st.slider(
                        "Maximum depth",
                        min_value=1,
                        max_value=20,
                        value=3,
                        key="gb_max_depth"
                    )

                    if framework == "xgboost":
                        subsample = st.slider(
                            "Subsample ratio",
                            min_value=0.1,
                            max_value=1.0,
                            value=1.0,
                            key="xgb_subsample"
                        )
                        colsample_bytree = st.slider(
                            "Column sample by tree",
                            min_value=0.1,
                            max_value=1.0,
                            value=1.0,
                            key="xgb_colsample"
                        )
                        model_params = {
                            'framework': 'xgboost',
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree
                        }

                    elif framework == "lightgbm":
                        num_leaves = st.slider(
                            "Number of leaves",
                            min_value=2,
                            max_value=256,
                            value=31,
                            key="lgb_num_leaves"
                        )
                        model_params = {
                            'framework': 'lightgbm',
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'num_leaves': num_leaves
                        }

                    elif framework == "catboost":
                        l2_leaf_reg = st.slider(
                            "L2 regularization",
                            min_value=1,
                            max_value=10,
                            value=3,
                            key="cat_l2_leaf_reg"
                        )
                        model_params = {
                            'framework': 'catboost',
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'l2_leaf_reg': l2_leaf_reg
                        }

                    else:  # sklearn
                        subsample = st.slider(
                            "Subsample ratio",
                            min_value=0.1,
                            max_value=1.0,
                            value=1.0,
                            key="gb_subsample"
                        )
                        model_params = {
                            'framework': 'sklearn',
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'subsample': subsample
                        }

                else:
                    # ==========
                    # Auto-Tuning
                    # ==========

                    st.info("Auto-tuning will try multiple parameter combinations to find the best model. "
                            "This may take a few minutes.")

                    # Instead of collecting manual params, we just flag that we want auto-tune
                    # plus store which framework we chose
                    model_params = {
                        'auto_tune': True, 
                        'framework': framework
                    }


            # In the modeling section, after other model types
            elif model_type == "lstm":
                # LSTM specific parameters
                timesteps = st.slider(
                    "Time Steps",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="Number of time steps to consider for sequence",
                    key="lstm_timesteps"
                )
                
                units = st.slider(
                    "LSTM Units",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of LSTM units in each layer",
                    key="lstm_units"
                )
                
                layers = st.slider(
                    "Number of LSTM Layers",
                    min_value=1,
                    max_value=3,
                    value=1,
                    help="Number of LSTM layers in the model",
                    key="lstm_layers"
                )
                
                dropout = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.1,
                    help="Dropout rate for regularization",
                    key="lstm_dropout"
                )
                
                epochs = st.slider(
                    "Training Epochs",
                    min_value=10,
                    max_value=200,
                    value=100,
                    step=10,
                    help="Number of training epochs",
                    key="lstm_epochs"
                )
                
                batch_size = st.slider(
                    "Batch Size",
                    min_value=16,
                    max_value=128,
                    value=32,
                    step=16,
                    help="Training batch size",
                    key="lstm_batch_size"
                )
                
                model_params = {
                    'timesteps': timesteps,
                    'units': units,
                    'layers': layers,
                    'dropout': dropout,
                    'epochs': epochs,
                    'batch_size': batch_size
                }

            # Add SVM-specific parameters
            elif model_type == "svm":
                kernel = st.selectbox(
                    "Kernel",
                    ["rbf", "linear", "poly", "sigmoid"],
                    help="The kernel type to be used in the algorithm",
                    key="svm_kernel"
                )
                
                C = st.slider(
                    "C (Regularization)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Regularization parameter. The strength of the regularization is inversely proportional to C",
                    key="svm_c"
                )
                
                gamma = st.selectbox(
                    "Gamma",
                    ["scale", "auto"] + [str(0.1 * i) for i in range(1, 11)],
                    help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'",
                    key="svm_gamma"
                )
                
                if kernel == "poly":
                    degree = st.slider(
                        "Polynomial Degree",
                        min_value=2,
                        max_value=5,
                        value=3,
                        help="Degree of polynomial kernel function",
                        key="svm_degree"
                    )
                    model_params = {
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma if gamma in ['scale', 'auto'] else float(gamma),
                        'degree': degree
                    }
                else:
                    model_params = {
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma if gamma in ['scale', 'auto'] else float(gamma)
                    }

            # Add TCN-specific parameters
            elif model_type == "tcn":
                sequence_length = st.slider(
                    "Sequence Length",
                    min_value=2,
                    max_value=50,
                    value=10,
                    help="Length of input sequences",
                    key="tcn_sequence_length"
                )
                
                nb_filters = st.slider(
                    "Number of Filters",
                    min_value=16,
                    max_value=128,
                    value=64,
                    step=16,
                    help="Number of filters in convolutional layers",
                    key="tcn_nb_filters"
                )
                
                kernel_size = st.slider(
                    "Kernel Size",
                    min_value=2,
                    max_value=8,
                    value=3,
                    help="Size of the convolutional kernel",
                    key="tcn_kernel_size"
                )
                
                nb_stacks = st.slider(
                    "Number of Stacks",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="Number of stacks of residual blocks",
                    key="tcn_nb_stacks"
                )
                
                max_dilation = st.slider(
                    "Maximum Dilation",
                    min_value=1,
                    max_value=32,
                    value=8,
                    help="Maximum dilation factor (powers of 2 will be used up to this value)",
                    key="tcn_max_dilation"
                )
                
                dropout_rate = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.1,
                    help="Dropout rate for regularization",
                    key="tcn_dropout"
                )
                
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.001,
                    format="%.4f",
                    help="Learning rate for optimization",
                    key="tcn_learning_rate"
                )
                
                epochs = st.slider(
                    "Training Epochs",
                    min_value=10,
                    max_value=200,
                    value=100,
                    step=10,
                    help="Number of training epochs",
                    key="tcn_epochs"
                )
                
                batch_size = st.slider(
                    "Batch Size",
                    min_value=16,
                    max_value=128,
                    value=32,
                    step=16,
                    help="Training batch size",
                    key="tcn_batch_size"
                )
                
                # Calculate dilations based on max_dilation
                dilations = [2**i for i in range(int(np.log2(max_dilation)) + 1)]
                
                model_params = {
                    'sequence_length': sequence_length,
                    'nb_filters': nb_filters,
                    'kernel_size': kernel_size,
                    'nb_stacks': nb_stacks,
                    'dilations': dilations,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size
                }

            # Add some spacing
            st.write("")
            
            

            if st.button("Train & Evaluate Model", key="train_model_btn"):
                
                if 'data_split_applied' not in st.session_state:
                    st.error("Please apply data split first before training the model.")
                    st.stop()
                    
                # Get the split data from session state
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
                X_val = st.session_state.get('X_val')  # Use get() to handle cases without validation
                y_val = st.session_state.get('y_val')
            
                with st.spinner('Training model...'):
                    
                    if model_type == "decision_tree":
                        if model_params.get('auto_tune'):
                            model_results = train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, model_params)
                        else:
                            model_results = train_model(
                                X_train, y_train,
                                X_test, y_test,
                                X_val, y_val,
                                model_type=model_type,
                                **model_params
                            )
                    
                    elif model_type == "random_forest":
                        if model_params.get('auto_tune'):
                            model_results = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, model_params)
                        else:
                            model_results = train_model(
                                X_train, y_train,
                                X_test, y_test,
                                X_val, y_val,
                                model_type=model_type,
                                **model_params
                            )
                        
                    elif model_type == "gradient_boost":
                        if model_params.get('auto_tune'):
                            model_results = train_and_evaluate_gradient_boost(X_train, y_train, X_test, y_test, X_val, y_val, model_params)
                            
                        else: 
                            model_results = train_model(
                                X_train, y_train,
                                X_test, y_test,
                                X_val, y_val,
                                model_type=model_type,
                                **model_params
                            )
                            
                    else:
                        # Original training code
                        model_results = train_model(
                            X_train, y_train,
                            X_test, y_test,
                            X_val, y_val,
                            model_type=model_type,
                            **model_params
                        )
                        
                    # Create main layout columns
                    metrics_col, plots_col = st.columns([1, 2])
                    
                    # Left column for metrics
                    with metrics_col:
                        st.markdown("### Model Performance")
                        st.markdown("#### Test Set Metrics")
                        metrics_df = pd.DataFrame(
                            model_results['test_metrics'].items(),
                            columns=['Metric', 'Value']
                        )
                        st.dataframe(metrics_df, hide_index=True)
                        
                        if model_results['val_metrics'] is not None:
                            st.markdown("#### Validation Set Metrics")
                            val_metrics_df = pd.DataFrame(
                                model_results['val_metrics'].items(),
                                columns=['Metric', 'Value']
                            )
                            st.dataframe(val_metrics_df, hide_index=True)

                    # Right column for plots
                    with plots_col:
                        # Custom CSS for plot containers
                        st.markdown("""
                            <style>
                                .plot-container {
                                    display: flex;
                                    justify-content: center;
                                    margin-bottom: 2rem;
                                }
                            </style>
                        """, unsafe_allow_html=True)

                        # Test Set Visualizations
                        st.markdown("#### Test Set Visualizations")
                        
                        # Confusion Matrix
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        fig_cm = plot_confusion_matrix(
                            y_test, 
                            model_results['test_predictions']
                        )
                        st.pyplot(fig_cm)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Feature Importance
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        fig_imp = plot_feature_importances(
                            model_results['feature_importances']
                        )
                        st.pyplot(fig_imp)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # ROC Curve
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        fig_roc = plot_roc_curve(
                            y_test,
                            model_results['test_predictions_proba']
                        )
                        st.pyplot(fig_roc)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Validation Set Visualizations (if available)
                        if model_results['val_metrics'] is not None:
                            st.markdown("#### Validation Set Visualizations")
                            
                            # Validation Confusion Matrix
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                            fig_val_cm = plot_confusion_matrix(
                                y_val,
                                model_results['val_predictions']
                            )
                            st.pyplot(fig_val_cm)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Validation ROC Curve
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                            fig_val_roc = plot_roc_curve(
                                y_val,
                                model_results['val_predictions_proba']
                            )
                            st.pyplot(fig_val_roc)
                            st.markdown('</div>', unsafe_allow_html=True)

                    # Store model results in session state
                    st.session_state['model_results'] = model_results
                    
                    # After model training, store feature names
                    st.session_state['model_results'] = model_results
                    st.session_state['model_metrics'] = metrics_df
                    st.session_state['val_metrics'] = val_metrics_df if model_results['val_metrics'] is not None else None
                    st.session_state['model_type'] = model_type
                    st.session_state['model_params'] = model_params
                    st.session_state['feature_names'] = X_train.columns.tolist()  # Add this line
                    st.session_state['model_trained'] = True
                    st.session_state['initial_training_done'] = True

                    
                    st.success('Model training complete!')
            
            #two columns for save and load buttons
            save_col, save_model_col = st.columns(2)
            
            with save_col:
            #save session keys as a json file using a save button
                if st.button("Save Session Data", key="save_session_btn"):
                    with st.spinner("Saving session data..."):
                        try:
                            saved_path = save_session_data(st.session_state)
                            st.success(f"Session data saved to {saved_path}")
                        except Exception as e:
                            st.error(f"Error saving session data: {str(e)}")
            
            with save_model_col:
                if st.button("Save Model"):
                    with st.spinner("Saving model..."):
                        try:
                            if 'model_results' not in st.session_state:
                                st.error("No trained model found. Please train a model first.")
                                return
                            
                            model_results = st.session_state['model_results']
                            model_type = st.session_state.get('model_type', 'unknown')
                            model_params = st.session_state.get('model_params', {})
                            feature_names = st.session_state.get('feature_names', [])  # Get from session state
                            
                            save_path = save_trained_model(
                                model_results,
                                model_type,
                                model_params,
                                feature_names
                            )
                            st.success(f"Model saved to {save_path}")
                            try:
                                saved_path = save_session_data(st.session_state)
                                st.success(f"Session data saved to {saved_path}")
                            except Exception as e:
                                st.error(f"Error saving session data: {str(e)}")
                

                        except Exception as e:
                            st.error(f"Error saving model: {str(e)}")
                        
        with model_tab:
            # Model information section
            st.markdown("### Available Models")
            
            # Display model cards in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Tree-based Models")
                with st.expander("Decision Tree"):
                    st.markdown("""
                        **Description**: Simple tree-based method that splits data on feature values.
                        
                        **Strengths**:
                        - Highly interpretable
                        - Handles non-linear relationships
                        - No feature scaling needed
                        
                        **Key Parameters**:
                        - max_depth
                        - min_samples_split
                        
                        **Feature Engineering Compatibility**:
                        - Rolling features: Good - can capture temporal patterns
                        - Lagging features: Good - can learn from historical values
                        - Best with moderate feature sets to maintain interpretability
                        
                        **Best For**: Small to medium datasets, when interpretability is crucial
                    """)
                
                with st.expander("Random Forest"):
                    st.markdown("""
                        **Description**: Ensemble of Decision Trees using bagging.
                        
                        **Strengths**:
                        - Better generalization than single trees
                        - Robust to overfitting
                        - Feature importance rankings
                        
                        **Key Parameters**:
                        - n_estimators
                        - max_depth
                        - min_samples_split
                        
                        **Feature Engineering Compatibility**:
                        - Rolling features: Excellent - can handle large feature sets
                        - Lagging features: Excellent - effectively combines historical patterns
                        - Can handle high-dimensional feature spaces well
                        
                        **Best For**: General-purpose classification, handling complex relationships
                    """)
                
                with st.expander("Gradient Boosting"):
                    st.markdown("""
                        **Description**: Sequential ensemble method that corrects previous errors.
                        
                        **Frameworks**:
                        - XGBoost
                        - LightGBM
                        - CatBoost
                        
                        **Key Parameters**:
                        - learning_rate
                        - n_estimators
                        - max_depth
                        
                        **Feature Engineering Compatibility**:
                        - Rolling features: Excellent - particularly good with statistical aggregations
                        - Lagging features: Excellent - can capture complex temporal dependencies
                        - Handles high-dimensional feature spaces efficiently
                        
                        **Best For**: When you need state-of-the-art performance on structured data
                    """)

            with col2:
                st.markdown("#### Other Models")
                with st.expander("Logistic Regression"):
                    st.markdown("""
                        **Description**: Linear model for classification.
                        
                        **Strengths**:
                        - Simple and interpretable
                        - Fast training
                        - Good baseline
                        
                        **Key Parameters**:
                        - C (regularization)
                        - solver
                        - max_iter
                        
                        **Feature Engineering Compatibility**:
                        - Rolling features: Limited - best with simple statistical aggregations
                        - Lagging features: Moderate - can use basic time-shifted features
                        - Feature selection important to prevent multicollinearity
                        
                        **Best For**: Linear problems, baseline model
                    """)
                
                with st.expander("Support Vector Machine"):
                    st.markdown("""
                        **Description**: Finds optimal hyperplane for classification.
                        
                        **Strengths**:
                        - Effective in high-dimensional spaces
                        - Versatile through different kernels
                        
                        **Key Parameters**:
                        - kernel
                        - C
                        - gamma
                        
                        **Feature Engineering Compatibility**:
                        - Rolling features: Good - especially with RBF kernel
                        - Lagging features: Good - but requires careful feature scaling
                        - Feature selection recommended for computational efficiency
                                
                        **Best For**: Medium-sized datasets, complex decision boundaries
                    """)
                
                with st.expander("Neural Networks (LSTM/TCN)"):
                    st.markdown("""
                        **Description**: Deep learning models for sequential data.
                        
                        **Types**:
                        - LSTM: Long Short-Term Memory
                        - TCN: Temporal Convolutional Network
                        
                        **Key Parameters**:
                        - layers/units
                        - sequence length
                        - learning rate
                        
                        **Feature Engineering Compatibility**:
                        - Rolling features: Optional - models can learn temporal patterns internally
                        - Lagging features: Usually unnecessary - built-in sequence handling
                        - Best with minimal feature engineering, raw sequential data
                        
                        **Best For**: Large datasets with temporal patterns
                    """)
                    
                    
            
        st.write("\n\n")
        st.write("--------------------------------")
        
        
        # Section 6: Model Prediction
        st.subheader("6. Model Prediction")

        #adding two tabs for prediction and model performance
        prediction_tab, model_performance_tab = st.tabs(["Prediction", "Model Performance"])

        with prediction_tab:
            # Get the list of saved models
            saved_models = get_saved_models()

            # Use a selectbox to choose the model to load
            model_to_load = st.selectbox("Select Model to Load", saved_models)

            # Load the model
            if st.button("Load Model", key="load_model_btn"):
                with st.spinner("Loading model..."):
                    try:
                        loaded_model = load_trained_model(model_to_load)
                        st.session_state['loaded_model'] = loaded_model
                        st.success(f"Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")

            # Update prediction interface in app
            if 'loaded_model' in st.session_state:
                # Use the transformed DataFrame from the session state
                transformed_df = st.session_state['transformed_df']

                # Extend the time series data using Monte Carlo simulation
                test_df = create_test_df(transformed_df)

                st.write("### Extended Time Series Data:")

                # Last 10 rows of the extended dataframe
                st.dataframe(test_df.tail(10))

                # Remove non-feature columns for prediction
                prediction_input = test_df.drop(columns=['Y'], errors='ignore')

                try:
                    results = make_predictions(st.session_state['loaded_model']['model'], prediction_input, test_df)

                    # Display results
                    st.write("### Predictions and Recommendations")
                    st.dataframe(results)
                    
                    # Information Section (Disclaimer and Explanation)
                    st.write("## 📘 Information and Disclaimer")
                    st.info(
                        """
                        **Disclaimer:**
                        - The data used for predictions is **synthetically generated** using a Monte Carlo simulation. It is not real market data.
                        - The predictions and recommendations provided are based on a machine learning model and should not be considered as financial advice.
                        - This tool is designed to help you **train your model, mitigate crash risks**, and **avoid potential losses** by making informed decisions in case of a market crash.
                        - Always conduct your own research or consult a financial advisor before making any investment decisions.

                        **AI-Powered Chatbot:**
                        - You can also use our **AI-powered chatbot** to help with **market research, understanding stock trends**, and getting answers to your questions about **investment strategies and market conditions**.

                        **Explanation of Metrics:**
                        - **Prediction**: The model predicts whether a market crash (1) or no crash (0) is likely to occur.
                        - **Recommendation**: Based on the prediction and confidence score, the model suggests whether to **Buy**, **Sell**, or **Hold**.
                        - **Confidence Score**: The model's confidence in its prediction. A higher confidence score indicates greater certainty.
                        - **Potential Return (%)**: The estimated return from following the recommendation. Assumed returns:
                            - **Buy**: +5% return if the market does not crash.
                            - **Sell**: +2% return if the market crashes.
                            - **Hold**: -1% return if the recommendation is incorrect or uncertain.
                        - **Cumulative Return (%)**: The total return accumulated over time by following the model’s recommendations. This metric helps track how an investment strategy would perform if followed consistently.
                        """
                    )

                    
                    # Plot cumulative return over time
                    fig, ax = plt.subplots()
                    ax.plot(results['Data'], results['Cumulative Return (%)'])
                    ax.set_title('Cumulative Return Over Time')
                    ax.set_xlabel('Data')
                    ax.set_ylabel('Cumulative Return (%)')
                    st.pyplot(fig)

                    
                    # Add save predictions button
                    if st.button("Save Predictions"):
                        try:
                            # Create predictions directory if it doesn't exist
                            os.makedirs('predictions', exist_ok=True)
                            
                            # Create filename with timestamp
                            output_filename = f"predictions/pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                            # Save predictions
                            results.to_csv(output_filename, index=False)
                            st.success(f"Predictions saved to {output_filename}")
                        except Exception as e:
                            st.error(f"Error saving predictions: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    
        with model_performance_tab:
            saved_models_data = get_models_with_metadata()

            if not saved_models_data:
                st.warning("No saved models found.")
            else:
                model_names = list(saved_models_data.keys())
                selected_model = st.selectbox(
                    "Select Model to View Performance",
                    model_names,
                    key="model_performance_select"
                )

                if selected_model:
                    metadata = saved_models_data[selected_model].get('metadata', {})

                    # Model Overview
                    st.markdown("### Model Overview")
                    col1, col2 = st.columns(2)

                    # Safely extract model type and training date
                    model_type = metadata.get('model_performance', {}).get('model_type', 'N/A')
                    training_date = metadata.get('metadata', {}).get('timestamp', 'N/A')

                    with col1:
                        st.write("**Model Type:**", model_type)
                        st.write("**Training Date:**", training_date)

                    # Safely extract features used and framework
                    feature_names = metadata.get('saved_keys', {}).get('feature_names', {}).get('value', [])
                    framework = metadata.get('model_performance', {}).get('model_params', {}).get('framework', 'N/A')

                    with col2:
                        st.write("**Features Used:**", len(feature_names) if feature_names else "N/A")
                        st.write("**Framework:**", framework)

                    # Model Parameters
                    with st.expander("Model Parameters"):
                        model_params = metadata.get('model_performance', {}).get('model_params', {})
                        st.json(model_params if model_params else {"message": "No parameters available"})

                    # Performance Metrics
                    with st.expander("Performance Metrics"):
                        test_metrics = metadata.get('model_performance', {}).get('test_metrics', {})
                        if test_metrics:
                            st.markdown("#### Test Set Metrics")
                            metrics_df = pd.DataFrame({
                                'Metric': list(test_metrics.keys()),
                                'Value': list(test_metrics.values())
                            })
                            st.dataframe(metrics_df)
                        else:
                            st.write("No test metrics available.")

                    # Feature Information
                    with st.expander("Feature Information"):
                        selected_features = metadata.get('saved_keys', {}).get('selected_features', {}).get('value', [])
                        st.markdown("#### Selected Features")
                        st.write(selected_features if selected_features else "No features selected.")

                        st.markdown("#### Feature Correlations")
                        try:
                            corr_str = metadata.get('saved_keys', {}).get('correlations', {}).get('value', "")
                            correlations = pd.Series({
                                k: float(v) for k, v in [
                                    line.split() for line in corr_str.split('\n') if line
                                ]
                            })
                            st.dataframe(correlations.sort_values(ascending=False))
                        except Exception as e:
                            st.error(f"Error displaying correlations: {str(e)}")



# Function to predict crash or no crash with confidence scores and propose an investment strategy
def make_predictions(model, input_df, copy_df, buy_threshold=0.7, sell_threshold=0.7):

    # Drop 'Data' column for prediction
    features = input_df.drop(columns=['Data'], errors='ignore')

    # Make predictions
    predictions = model.predict(features)

    # Get confidence scores using predict_proba
    if hasattr(model, 'predict_proba'):
        prob_scores = model.predict_proba(features)
        confidence_scores = prob_scores.max(axis=1)  # Get the max probability for each prediction
    else:
        confidence_scores = [None] * len(predictions)  # Handle models without predict_proba

    # Use the copy DataFrame to add predictions and recommendations
    copy_df['Prediction'] = predictions
    copy_df['Confidence Score'] = confidence_scores

    # Propose investment recommendations based on confidence thresholds
    recommendations = []
    returns = []

    for pred, conf in zip(predictions, confidence_scores):
        if pred == 0 and conf >= buy_threshold:
            recommendations.append("Buy")
            returns.append(5)  # Assume 5% return for correct Buy decision
        elif pred == 1 and conf >= sell_threshold:
            recommendations.append("Sell")
            returns.append(2)  # Assume 2% return for correct Sell decision
        else:
            recommendations.append("Hold")
            returns.append(-1)  # Assume -1% loss for incorrect decision

    # Add recommendations and returns to the DataFrame
    copy_df['Recommendation'] = recommendations
    copy_df['Potential Return (%)'] = returns

    # Calculate cumulative returns over time
    copy_df['Cumulative Return (%)'] = copy_df['Potential Return (%)'].cumsum()

    # Return the enhanced DataFrame
    return copy_df[['Data', 'Prediction', 'Confidence Score', 'Recommendation', 'Potential Return (%)', 'Cumulative Return (%)']]



# Function to load a trained model
def load_trained_model(model_path):
    """Load a trained model and metadata from disk."""
    try:
        # Construct full path to model file
        full_path = os.path.join('models', model_path, 'model.joblib')

        # Check if file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found at: {full_path}")

        # Load model data using joblib
        model_data = joblib.load(full_path)
        return model_data

    except Exception as e:
        raise Exception(f"Error loading model from {full_path}: {str(e)}")


# Function to create a new time series test DataFrame using Monte Carlo Simulation
def create_test_df(dataframe, start_date="2021-01-01", end_date="2025-12-31"):
    """
    Create a new time series test DataFrame using Monte Carlo simulation for the given date range.
    """
    # Convert 'Data' column to datetime
    dataframe['Data'] = pd.to_datetime(dataframe['Data'])

    # Generate future dates for the test DataFrame
    future_dates = pd.date_range(start=start_date, end=end_date, freq='W')

    # Initialize a new DataFrame for test data
    test_df = pd.DataFrame({'Data': future_dates})

    # For each feature, simulate future values using random walk
    for col in dataframe.columns:
        if col not in ['Data', 'Y']:  # Skip target column and date column
            historical_values = dataframe[col].values
            simulated_values = np.random.normal(
                loc=np.mean(historical_values),
                scale=np.std(historical_values),
                size=len(future_dates)
            )
            # Add simulated values to the test DataFrame
            test_df[col] = simulated_values

    # Fill any missing values using forward fill
    test_df.fillna(method='ffill', inplace=True)

    return test_df


def save_trained_model(model_results, model_type, model_params, feature_names):
    
    if not model_results or not model_type:
        raise ValueError("Missing required model data")
        
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create a save path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'models/{model_type}_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and metadata
    model_data = {
        'model': model_results['model'],
        'feature_names': feature_names,
        'model_type': model_type,
        'model_params': model_params,
        'metrics': {
            'test_metrics': model_results['test_metrics'],
            'val_metrics': model_results['val_metrics']
        }
    }
    
    joblib.dump(model_data, f'{model_dir}/model.joblib')
    return model_dir

#get available models from the models directory
def get_saved_models():
    models = []
    for root, dirs, files in os.walk('models'):
        for dir in dirs:
            if os.path.exists(f'{root}/{dir}/model.joblib'):
                models.append(dir)
    return models
        

def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val, model_params):
    """Train and evaluate the Decision Tree model."""
    progress_text = st.empty()
    progress_text.text("Auto-tuning in progress... This may take a few minutes.")
    
    # Get top models from auto-tuning
    top_models = auto_tune_decision_tree(
        X_train, y_train,
        X_test, y_test,
        X_val=X_val, y_val=y_val
    )
    
    # Display top models in an expander
    with st.expander("Top 5 Models from Auto-Tuning"):
        for i, result in enumerate(top_models, 1):
            st.markdown(f"**Model {i}**")
            st.write(f"Parameters: {result['params']}")
            st.write(f"Overall Score: {result['overall_score']:.4f}")
            st.write("Metrics:", result['metrics'])
            st.write("---")
    
    # Use the best model's results for visualization
    model_results = top_models[0]['model_results']
    
    test_metrics = get_performance_metrics(y_test, model_results['test_predictions'])
    test_metrics['CV Score Mean'] = model_results['cv_scores'].mean()
    test_metrics['CV Score Std'] = model_results['cv_scores'].std()
    
    val_metrics = None
    if X_val is not None and y_val is not None and model_results['val_predictions'] is not None:
        val_metrics = get_performance_metrics(y_val, model_results['val_predictions'])
    
    model_results['test_metrics'] = test_metrics
    model_results['val_metrics'] = val_metrics
    
    progress_text.empty()    
    
    return model_results

def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test, X_val, y_val, model_params):
    """Train and evaluate the Random Forest model."""
    progress_text = st.empty()
    progress_text.text("Auto-tuning in progress... This may take a few minutes.")
    
    # Get top models from auto-tuning
    top_models = auto_tune_random_forest(
        X_train, y_train,
        X_test, y_test,
        X_val=X_val, y_val=y_val
    )
    
    # Display top models in an expander
    with st.expander("Top 5 Models from Auto-Tuning"):
        for i, result in enumerate(top_models, 1):
            st.markdown(f"**Model {i}**")
            st.write(f"Parameters: {result['params']}")
            st.write(f"Overall Score: {result['overall_score']:.4f}")
            st.write("Metrics:", result['metrics'])
            st.write("---")
    
    # Use the best model's results for visualization
    model_results = top_models[0]['model_results']
    
    test_metrics = get_performance_metrics(y_test, model_results['test_predictions'])
    test_metrics['CV Score Mean'] = model_results['cv_scores'].mean()
    test_metrics['CV Score Std'] = model_results['cv_scores'].std()
    
    val_metrics = None
    if X_val is not None and y_val is not None and model_results['val_predictions'] is not None:
        val_metrics = get_performance_metrics(y_val, model_results['val_predictions'])
    
    model_results['test_metrics'] = test_metrics
    model_results['val_metrics'] = val_metrics
    
    progress_text.empty()
    
    return model_results

def train_and_evaluate_gradient_boost(X_train, y_train, X_test, y_test, X_val, y_val, model_params):
    """Train and evaluate the Gradient Boosting model."""
    progress_text = st.empty()
    progress_text.text("Auto-tuning in progress... This may take a few minutes.")
    
    #get the framework from model_params
    framework = model_params.get('framework')
    
    print(f"FRAMEWORK: {framework}")
    
    # Get top models from auto-tuning
    top_models = auto_tune_gradient_boosting(
        X_train, y_train,
        X_test, y_test,
        X_val=X_val, y_val=y_val, framework=framework
    )
    
    # Display top models in an expander
    with st.expander("Top 5 Models from Auto-Tuning"):
        for i, result in enumerate(top_models, 1):
            st.markdown(f"**Model {i}**")
            st.write(f"Parameters: {result['params']}")
            st.write(f"Overall Score: {result['overall_score']:.4f}")
            st.write("Metrics:", result['metrics'])
            st.write("---")
    
    # Use the best model's results for visualization
    model_results = top_models[0]['model_results']
    
    test_metrics = get_performance_metrics(y_test, model_results['test_predictions'])
    test_metrics['CV Score Mean'] = model_results['cv_scores'].mean()
    test_metrics['CV Score Std'] = model_results['cv_scores'].std()
    
    val_metrics = None
    if X_val is not None and y_val is not None and model_results['val_predictions'] is not None:
        val_metrics = get_performance_metrics(y_val, model_results['val_predictions'])
    
    model_results['test_metrics'] = test_metrics
    model_results['val_metrics'] = val_metrics
    
    progress_text.empty()
    
    return model_results

def save_session_data(session_state, base_dir='./saved_models'):
    """Save session state data with full key-value pairs in timestamped folders"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(base_dir, f'model_{timestamp}')
    
    # Create subfolders including metrics
    data_dir = os.path.join(model_dir, 'data')
    model_artifacts_dir = os.path.join(model_dir, 'model_artifacts')
    plots_dir = os.path.join(model_dir, 'plots')
    metrics_dir = os.path.join(model_dir, 'metrics')
    
    for directory in [data_dir, model_artifacts_dir, plots_dir, metrics_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize session data with model performance section
    session_data = {
        'metadata': {
            'timestamp': timestamp,
            'session_id': str(uuid.uuid4())
        },
        'model_performance': {
            'test_metrics': {},
            'validation_metrics': None,
            'model_type': session_state.get('model_type'),
            'model_params': session_state.get('model_params')
        },
        'saved_keys': {}
    }

    # Save model metrics
    if 'model_metrics' in session_state:
        metrics_df = session_state['model_metrics']
        metrics_dict = dict(zip(metrics_df['Metric'], metrics_df['Value']))
        session_data['model_performance']['test_metrics'] = metrics_dict
        
        # Save metrics DataFrame
        metrics_path = os.path.join(metrics_dir, f'test_metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
    if 'val_metrics' in session_state and session_state['val_metrics'] is not None:
        val_metrics_df = session_state['val_metrics']
        val_metrics_dict = dict(zip(val_metrics_df['Metric'], val_metrics_df['Value']))
        session_data['model_performance']['validation_metrics'] = val_metrics_dict
        
        # Save validation metrics DataFrame
        val_metrics_path = os.path.join(metrics_dir, f'validation_metrics_{timestamp}.csv')
        val_metrics_df.to_csv(val_metrics_path, index=False)

    # Rest of existing save logic
    for key, value in session_state.items():
        try:
            if isinstance(value, pd.DataFrame):
                df_path = os.path.join(data_dir, f'{key}_{timestamp}.csv')
                value.to_csv(df_path, index=False)
                session_data['saved_keys'][key] = {
                    'type': 'DataFrame',
                    'path': df_path,
                    'shape': value.shape
                }
            
            elif isinstance(value, (plt.Figure, dict)):
                # Save plots and dictionaries
                if isinstance(value, plt.Figure):
                    fig_path = os.path.join(plots_dir, f'{key}_{timestamp}.png')
                    value.savefig(fig_path)
                    session_data['saved_keys'][key] = {
                        'type': 'Figure',
                        'path': fig_path
                    }
                else:
                    dict_path = os.path.join(model_artifacts_dir, f'{key}_{timestamp}.json')
                    with open(dict_path, 'w') as f:
                        json.dump(value, f, indent=2)
                    session_data['saved_keys'][key] = {
                        'type': 'Dictionary',
                        'path': dict_path
                    }
            
            elif hasattr(value, 'get_params'):
                # Handle sklearn models
                model_path = os.path.join(model_artifacts_dir, f'{key}_{timestamp}.joblib')
                from joblib import dump
                dump(value, model_path)
                session_data['saved_keys'][key] = {
                    'type': 'Model',
                    'path': model_path,
                    'params': value.get_params()
                }
            
            else:
                # Save basic types directly
                session_data['saved_keys'][key] = {
                    'type': str(type(value).__name__),
                    'value': value if isinstance(value, (int, float, str, bool, list)) else str(value)
                }

        except Exception as e:
            session_data['saved_keys'][key] = {
                'type': 'Error',
                'error_message': str(e)
            }

    # Save updated session metadata
    metadata_path = os.path.join(model_dir, f'session_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(session_data, f, indent=2)

    return model_dir
    
    # Save dataframes
    if 'df' in session_state:
        df_path = os.path.join(session_dir, f'df_{timestamp}.csv')
        session_state['df'].to_csv(df_path, index=False)
        session_data['df_path'] = df_path
        
    if 'transformed_df' in session_state:
        trans_path = os.path.join(session_dir, f'transformed_df_{timestamp}.csv')
        session_state['transformed_df'].to_csv(trans_path, index=False)
        session_data['transformed_df_path'] = trans_path
    
    # Save session state to JSON
    json_path = os.path.join(session_dir, f'session_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(session_data, f, indent=2)
        
    return json_path

def get_models_with_metadata():
    """Get all saved models with their metadata."""
    models_data = {}
    
    try:
        saved_models_path = './saved_models'
        if not os.path.exists(saved_models_path):
            print(f"Directory not found: {saved_models_path}")
            return {}
            
        # List model directories
        model_dirs = [d for d in os.listdir(saved_models_path) if os.path.isdir(os.path.join(saved_models_path, d))]
        
        for dir_name in model_dirs:
            model_dir = os.path.join(saved_models_path, dir_name)
            
            # Look for session_metadata file directly in model directory
            metadata_files = [f for f in os.listdir(model_dir) if 'session_metadata' in f and f.endswith('.json')]
            
            if metadata_files:
                metadata_path = os.path.join(model_dir, metadata_files[0])
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    models_data[dir_name] = {
                        'metadata': metadata,
                        'path': model_dir,
                        'timestamp': metadata['metadata'].get('timestamp', 'N/A')
                    }
                except Exception as e:
                    print(f"Error reading metadata for {dir_name}: {str(e)}")
                    continue
                    
        return models_data
        
    except Exception as e:
        print(f"Error accessing saved models: {str(e)}")
        return {}



if __name__ == "__main__":
    main()
