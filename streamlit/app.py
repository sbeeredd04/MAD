import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import helper functions from helper.py
from helper import (
    compute_correlations,
    create_time_split,
    load_and_analyze_csv,
    get_available_csvs, 
    get_feature_engineering_options,
    apply_feature_engineering,
    save_dataframe_to_csv,
    create_time_split_with_validation,
    train_model, 
    compute_permutation_importance,
    create_feature_importance_summary,
    plot_permutation_importance
)
from plotter import plot_model_results, plot_confusion_matrix, plot_feature_importances, plot_roc_curve

def main():
    
    #using full width for the title
    st.title("Anomaly (or Classification) Detection Model Builder")
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
            
    
    #spacer for the next section
    st.write("\n\n")
    st.write("--------------------------------")
    
    # 2. Feature Engineering
    st.subheader("2. Feature Engineering")

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

    # Get feature engineering options
    fe_options = get_feature_engineering_options()
    selected_method = st.selectbox(
        "Select Feature Engineering Method:",
        options=list(fe_options.keys()),
        help="Choose the method to transform your features"
    )

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
    selected_fe_cols = st.multiselect(
        "Select features to transform:",
        feature_cols,
        default=feature_cols
    )

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

    # Initialize parameters dictionary
    fe_params = {
        'target_column': target_col,
        'correlation_method': corr_method
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
            # Apply feature engineering
            transformed_df, feature_summary = apply_feature_engineering(
                df,
                selected_fe_cols,
                target_col=target_col,
                method=selected_method,
                params=fe_params
            )
            
            # Display results in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Transformed Data Preview")
                st.write(f"Shape: {transformed_df.shape}")
                st.dataframe(transformed_df.head(10))
            
            with col2:
                st.write("### Feature Correlations with Target")
                # Get correlations with target and sort by absolute value
                all_features = [col for col in transformed_df.columns 
                              if col not in [target_col, 'Data']]
                correlations = transformed_df[all_features].corrwith(
                    transformed_df[target_col], 
                    method=corr_method
                )
                correlations = correlations.abs().sort_values(ascending=False)
                
                # Display correlation values as a formatted dataframe
                correlation_data = pd.DataFrame({
                    'Feature': correlations.index,
                    'Correlation': correlations.values
                })
                st.dataframe(correlation_data)
            
            # Store transformed data and parameters in session state
            st.session_state.update({
                'transformed_df': transformed_df,
                'feature_summary': feature_summary,
                'target_column': target_col,
                'selected_features': all_features,  # Update to include new features
                'correlation_method': corr_method,
                'feature_correlations': correlation_data
            })
            
            st.success("Feature engineering applied successfully!")

            # Option to save the transformed data
            if st.button("Save Transformed Data"):
                fe_method_name = selected_method.lower().replace(' ', '_')
                if selected_method == 'Rolling Features':
                    fe_method_name = f'rolling_features_{fe_params["window_size"]}_{"-".join(fe_params["operations"])}'
                elif selected_method == 'Lagging Features':
                    fe_method_name = f'lagging_features_{"-".join(map(str, fe_params["lag_periods"]))}'
                
                save_dataframe_to_csv(
                    transformed_df, 
                    file_path='../data/transformed_data/', 
                    name=f'transformed_data_{fe_method_name}.csv'
                )
                st.success("Data saved successfully!")

    st.write("\n\n")
    st.write("--------------------------------")
    
    # 3. Feature Relevance
    st.subheader("3. Feature Relevance")

    # Check if feature engineering has been applied
    if 'transformed_df' not in st.session_state:
        st.warning("Please apply feature engineering first.")
        st.stop()

    # Get the transformed dataframe and target column
    df = st.session_state['transformed_df']
    target_col = st.session_state['target_column']

    # Add unique key to the slider
    corr_threshold = st.slider(
        "Correlation threshold (absolute value):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.05, 
        step=0.01,
        key="feature_relevance_threshold"
    )

    # Initialize selected_features if not in session state
    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = [
            col for col in df.columns 
            if col not in [target_col, 'Data']
        ]

    # Add unique key to the button
    if st.button("Compute Feature Relevance", key="compute_feature_relevance_btn"):
        with st.spinner("Computing correlations..."):
            # Compute correlations using the transformed data
            correlations, corr_dict = compute_correlations(
                df, 
                target_col=target_col, 
                threshold=corr_threshold
            )

            # Display results
            st.write("### All correlations with target (sorted by absolute value):")
            st.dataframe(correlations, key="correlations_table")

            st.write("### Selected features after correlation filtering:")
            st.write(corr_dict['selected_features'])

            # Update selected features in session state
            st.session_state['selected_features'] = corr_dict['selected_features']
            st.session_state['feature_relevance'] = correlations

            st.success("Feature relevance analysis complete!")

    # Get selected features for next steps
    selected_features = st.session_state['selected_features']

    # Now you can use selected_features in subsequent steps
    if st.button("Proceed to Model Training", key="proceed_to_model_btn"):
        # Create train/test split using selected features
        X_train, X_test, X_val, y_train, y_test, y_val = create_time_split(
            df, 
            target_col=target_col,
            selected_features=selected_features,  # Now this will be defined
            test_split_date=test_split_date,
            val_split_date=val_split_date if include_validation else None
        )
        
        # Store splits in session state
        st.session_state.update({
            'X_train': X_train,
            'X_test': X_test,
            'X_val': X_val,
            'y_train': y_train,
            'y_test': y_test,
            'y_val': y_val
        })

    #spacer for the next section
    st.write("\n\n")
    st.write("--------------------------------")
    


    # 4. Train/Test/Validation Split
    st.subheader("4. Train/Test/Validation Split")
    
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

    # Calculate split dates and create masks
    total_days = (max_date - min_date).days
    test_split_date = min_date + pd.Timedelta(days=int(st.session_state.splits['train'] * total_days / 100))
    
    if include_validation:
        val_split_date = test_split_date + pd.Timedelta(days=int(st.session_state.splits['test'] * total_days / 100))
    
    # Calculate and display sample counts
    train_mask = df['Data'] < test_split_date
    if include_validation:
        test_mask = (df['Data'] >= test_split_date) & (df['Data'] < val_split_date)
        val_mask = df['Data'] >= val_split_date
    else:
        test_mask = df['Data'] >= test_split_date
    
    # Display sample counts in horizontal layout
    cols = st.columns(3)
    with cols[0]:
        train_samples = len(df[train_mask])
        st.metric("Train Samples", f"{train_samples:,}")
    
    with cols[1]:
        test_samples = len(df[test_mask])
        st.metric("Test Samples", f"{test_samples:,}")
    
    with cols[2]:
        if include_validation:
            val_samples = len(df[val_mask])
            st.metric("Validation Samples", f"{val_samples:,}")
    
    # Store split data in session state
    if include_validation:
        X_train, X_test, X_val, y_train, y_test, y_val = create_time_split_with_validation(
            df, target_col='Y', selected_features=selected_features,
            test_split_date=test_split_date, val_split_date=val_split_date
        )
        st.session_state['X_val'] = X_val
        st.session_state['y_val'] = y_val
    else:
        X_train, X_test, y_train, y_test = create_time_split(
            df, target_col='Y', selected_features=selected_features,
            split_date=test_split_date
        )
    
    st.session_state.update({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'include_validation': include_validation
    })
    
    
    #spacer for the next section
    st.write("\n\n")
    st.write("--------------------------------")
    
    # 5. Modeling
    st.subheader("5. Model Training & Evaluation")

    # Add unique keys to all sliders
    model_type = st.selectbox("Model Type", 
                             ["decision_tree", "random_forest", "gradient_boost"],
                             key="model_type_select")


    if model_type == "decision_tree":
        max_depth_dt = st.slider("max_depth", 1, 20, 5, 1, key="dt_depth")
        min_samples_split_dt = st.slider("min_samples_split", 2, 20, 5, 1, key="dt_split")
        model_params = {
            'max_depth': max_depth_dt,
            'min_samples_split': min_samples_split_dt
        }
    elif model_type == "random_forest":
        n_estimators_rf = st.slider("n_estimators", 10, 300, 100, 10, key="rf_estimators")
        max_depth_rf = st.slider("max_depth", 1, 20, 10, 1, key="rf_depth")
        model_params = {
            'n_estimators': n_estimators_rf,
            'max_depth': max_depth_rf,
        }

    if model_type == "random_forest":
        min_samples_split_rf = st.slider("min_samples_split", 2, 20, 5, 1, key="rf_split")
        model_params['min_samples_split'] = min_samples_split_rf
    elif model_type == "gradient_boost":
        n_estimators_gb = st.slider("n_estimators", 10, 300, 100, 10, key="gb_estimators")
        learning_rate_gb = st.select_slider("learning_rate", 
                                            options=[0.01, 0.05, 0.1, 0.2], 
                                            value=0.1,
                                            key="gb_lr")
        max_depth_gb = st.slider("max_depth", 1, 10, 3, 1, key="gb_depth")
        model_params = {
            'n_estimators': n_estimators_gb,
            'learning_rate': learning_rate_gb,
            'max_depth': max_depth_gb
        }

    # Add some spacing
    st.write("")

    if st.button("Train & Evaluate Model", key="train_model_btn"):
        with st.spinner('Training model...'):
            # Train model with validation set if included
            if include_validation:
                model_results = train_model(
                    X_train, y_train,
                    X_test, y_test,
                    X_val, y_val,
                    model_type=model_type,
                    **model_params
                )
            else:
                model_results = train_model(
                    X_train, y_train,
                    X_test, y_test,
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
            st.session_state['initial_training_done'] = True
            
            st.success('Model training complete!')
    
    # # After initial training button
    # if 'initial_training_done' in st.session_state:
    #     model_results = st.session_state['model_results']
        
    #     st.write("\n")
    #     st.markdown("### Feature Selection & Retraining")
        
    #     # Get number of features
    #     n_features = len(model_results['feature_importances'])
        
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         n_top_features = st.slider(
    #             "Number of top features to keep",
    #             min_value=2,
    #             max_value=n_features,
    #             value=min(10, n_features),
    #             key="n_top_features"
    #         )
        
    #     with col2:
    #         n_permutations = st.slider(
    #             "Number of permutations for importance calculation",
    #             min_value=5,
    #             max_value=50,
    #             value=10,
    #             key="n_permutations"
    #         )

    #     if st.button("Compute Permutation Importance & Retrain", key="retrain_btn"):
    #         with st.spinner("Computing permutation importance..."):
    #             # Compute permutation importance
    #             perm_importance = compute_permutation_importance(
    #                 model_results['model'],
    #                 X_test,
    #                 y_test,
    #                 n_repeats=n_permutations
    #             )
                
    #             # Create feature importance summary
    #             importance_summary = create_feature_importance_summary(
    #                 model_results['feature_importances'],
    #                 perm_importance,
    #                 X_train.columns
    #             )
                
    #             # Store in session state
    #             st.session_state['importance_summary'] = importance_summary
                
    #             # Plot permutation importance
    #             fig_perm = plot_permutation_importance(importance_summary)
    #             st.pyplot(fig_perm)
                
    #             # Get top features
    #             top_features = importance_summary.index[:n_top_features].tolist()
                
    #             # Create filtered datasets
    #             X_train_filtered = X_train[top_features]
    #             X_test_filtered = X_test[top_features]
    #             X_val_filtered = X_val[top_features] if include_validation else None
                
    #             # Store filtered datasets in session state
    #             st.session_state.update({
    #                 'X_train_filtered': X_train_filtered,
    #                 'X_test_filtered': X_test_filtered,
    #                 'X_val_filtered': X_val_filtered,
    #                 'top_features': top_features
    #             })
                
    #             # Retrain model with filtered features
    #             with st.spinner('Retraining model with selected features...'):
    #                 if include_validation:
    #                     retrained_results = train_model(
    #                         X_train_filtered, y_train,
    #                         X_test_filtered, y_test,
    #                         X_val_filtered, y_val,
    #                         model_type=model_type,
    #                         **model_params
    #                     )
    #                 else:
    #                     retrained_results = train_model(
    #                         X_train_filtered, y_train,
    #                         X_test_filtered, y_test,
    #                         model_type=model_type,
    #                         **model_params
    #                     )
                    
    #                 # Store retrained results
    #                 st.session_state.update({
    #                     'retrained_model_results': retrained_results,
    #                     'retraining_done': True
    #                 })

    #     # Show results if retraining is done
    #     if 'retraining_done' in st.session_state:
    #         retrained_results = st.session_state['retrained_model_results']
    #         model_results = st.session_state['model_results']
            
    #         st.markdown("### Model Performance Comparison")
            
    #         # Create comparison dataframe
    #         comparison_data = {
    #             'Metric': [],
    #             'Original Model': [],
    #             'Retrained Model': []
    #         }
            
    #         for metric in model_results['test_metrics']:
    #             comparison_data['Metric'].append(metric)
    #             comparison_data['Original Model'].append(
    #                 model_results['test_metrics'][metric]
    #             )
    #             comparison_data['Retrained Model'].append(
    #                 retrained_results['test_metrics'][metric]
    #             )
            
    #         comparison_df = pd.DataFrame(comparison_data)
    #         st.session_state['comparison_df'] = comparison_df
    #         st.dataframe(comparison_df, hide_index=True)
            
    #         # Display visualizations
    #         metrics_col, plots_col = st.columns([1, 2])
            
    #         with plots_col:
    #             st.markdown("#### Retrained Model Visualizations")
                
    #             # Confusion Matrix
    #             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    #             fig_cm = plot_confusion_matrix(
    #                 y_test,
    #                 retrained_results['test_predictions']
    #             )
    #             st.pyplot(fig_cm)
    #             st.markdown('</div>', unsafe_allow_html=True)
                
    #             # ROC Curve
    #             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    #             fig_roc = plot_roc_curve(
    #                 y_test,
    #                 retrained_results['test_predictions_proba']
    #             )
    #             st.pyplot(fig_roc)
    #             st.markdown('</div>', unsafe_allow_html=True)
                        
    #             st.session_state['retraining_done'] = True
    #             st.success('Model retraining complete!')



if __name__ == "__main__":
    main()
