import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from IPython.display import display
def missing_summary(df, caption="Summary of Missing"):
    summary = (
        df.isnull().sum().to_frame('Missing Count')
        .assign(**{'Missing Percentage': lambda x: x['Missing Count'] / len(df) * 100})
    )
    summary.loc['Dataset Duplicates'] = [df.duplicated().sum(), np.nan]

    styled = (
        summary.style
        .set_caption(caption)
        .format({'Missing Count': '{:,.0f}', 'Missing Percentage': '{:.2f}%'})
        .set_properties(**{'background-color': '#1E1E1E', 'color': '#FFF', 'border': '1px solid #444'})
        .set_table_styles([{'selector': 'th', 'props': [('background-color', '#333'),
                                                       ('color', '#FFF'),
                                                       ('font-weight', 'bold')]}])
        .set_properties(subset=['Missing Count'], **{'color': '#ff7300'})
        .set_properties(subset=['Missing Percentage'], **{'color': '#b70000'})
    )
    return styled


def describe_styled(df, caption="Descriptive Statistics Summary"):
    desc = df.describe().T
    colors = {'count':'#b70000','mean':'#ff7300','std':'#ffdf00','min':'#86b412',
              '25%':'#3d8420','50%':'#0672b0','75%':'#0d896','max':'#a6017e'}
    fmt = {c:'{:,.4f}' for c in desc.columns}; fmt['count']='{:,.0f}'

    styled = (desc.style
              .set_properties(**{'background-color':'#1E1E1E','color':'#FFF','border':'1px solid #444'})
              .set_table_styles([{'selector':'th','props':[('background-color','#333'),
                                                           ('color','#FFF'),
                                                           ('font-weight','bold')]}])
              .set_caption(caption))
    for c,k in colors.items():
        if c in desc.columns: styled = styled.set_properties(subset=[c], **{'color':k})
    return styled


def missing_duplicate_heatmap(df):
    print(f'Missing values: {df.isnull().sum().sum()}')
    print(f'Duplicate values: {df.duplicated().sum()}')
    
    sns.set_style("darkgrid")
    plt.figure(figsize=(14,5))
    
    plt.subplot(1,2,1)
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=14, color='#ff7300')
    
    plt.subplot(1,2,2)
    sns.heatmap(df.duplicated().values.reshape(-1,1), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Duplicate Values Heatmap', fontsize=14, color='#b70000')
    
    plt.tight_layout()
    plt.show()

def skewness_styled(df, numeric_cols=None, caption="Skewness Analysis"):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    def classify_skewness(skew_value):
        if skew_value > 1: return "Highly Right Skewed"
        elif skew_value > 0.5: return "Moderately Right Skewed"
        elif skew_value < -1: return "Highly Left Skewed"
        elif skew_value < -0.5: return "Moderately Left Skewed"
        else: return "No Skew"
    
    
    skew_df = df[numeric_cols].skew().reset_index()
    skew_df.columns = ['Column Name', 'Skew Value']
    skew_df['Skew Type'] = skew_df['Skew Value'].apply(classify_skewness)
    
    
    colors = {
        "Highly Right Skewed": "#ff4d4d",
        "Moderately Right Skewed": "#ff9900",
        "No Skew": "#b7ff4d",
        "Moderately Left Skewed": "#66ccff",
        "Highly Left Skewed": "#9933ff"
    }
    

    def highlight_skew_type(row):
        color = colors.get(row['Skew Type'], "#1E1E1E")
        return ['background-color: {}; color: black; font-weight: bold;'.format(color) 
                if col == 'Skew Type' else '' for col in row.index]

    styled = (
        skew_df.style
        .set_caption(caption)
        .set_properties(**{'background-color':'#1E1E1E','color':'#FFF','border':'1px solid #444'})
        .set_table_styles([{'selector':'th','props':[('background-color','#333'),
                                                     ('color','#FFF'),
                                                     ('font-weight','bold')]}])
        .format({'Skew Value':'{:.4f}'})
        .apply(highlight_skew_type, axis=1)  # ✅ apply background properly
    )
    
    return styled


# display(skewness_styled(data, numeric_cols=numeric_cols_all))


def hist_with_skewness(df, numeric_cols=None, cols_plot=4, palette="husl"):
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    num_cols = len(numeric_cols)
    colors = sns.color_palette(palette, num_cols)
    rows_plot = (num_cols + cols_plot - 1) // cols_plot

    
    def classify_skewness(skew_value):
        if skew_value > 1: return "Highly Right Skewed"
        elif skew_value > 0.5: return "Moderately Right Skewed"
        elif skew_value < -1: return "Highly Left Skewed"
        elif skew_value < -0.5: return "Moderately Left Skewed"
        else: return "No Skew"


    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(20, rows_plot * 4))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_cols):
        skew_val = df[feature].skew()
        skew_type = classify_skewness(skew_val)
        
        sns.histplot(df[feature], ax=axes[i], color=colors[i % len(colors)], kde=True)
        axes[i].set_title(f"{feature} → {skew_type}\n(Skew={skew_val:.2f})", fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def outliers_styled(df, numeric_cols=None, caption="Outlier Analysis"):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    def classify_outlier(pct):
        if pct == 0: return "No Outliers"
        elif pct < 1: return "Very Few Outliers"
        elif pct < 5: return "Moderate Outliers"
        elif pct < 15: return "High Outliers"
        else: return "Severe Outliers"
    
    summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        pct = (len(outliers) / len(df)) * 100
        
        summary.append({
            "Column Name": col,
            "Outlier Count": len(outliers),
            "Total Count": len(df),
            "Outlier %": pct,
            "Outlier Type": classify_outlier(pct)
        })
    
    outlier_df = pd.DataFrame(summary)
    
    
    colors = {
        "No Outliers": "#b7ff4d",
        "Very Few Outliers": "#66ccff",
        "Moderate Outliers": "#ffcc00",
        "High Outliers": "#ff9900",
        "Severe Outliers": "#ff4d4d"
    }
    
    
    def highlight_outlier_type(row):
        color = colors.get(row['Outlier Type'], "#1E1E1E")
        return ['background-color: {}; color: black; font-weight: bold;'.format(color) 
                if col == 'Outlier Type' else '' for col in row.index]

    styled = (
        outlier_df.style
        .set_caption(caption)
        .set_properties(**{'background-color':'#1E1E1E','color':'#FFF','border':'1px solid #444'})
        .set_table_styles([{'selector':'th','props':[('background-color','#333'),
                                                     ('color','#FFF'),
                                                     ('font-weight','bold')]}])
        .format({'Outlier %':'{:.2f}%'})
        .apply(highlight_outlier_type, axis=1)
    )
    
    return styled



def box_with_outliers(df, numeric_cols=None, cols_plot=4, palette="husl"):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    num_cols = len(numeric_cols)
    colors = sns.color_palette(palette, num_cols)
    rows_plot = (num_cols + cols_plot - 1) // cols_plot

    
    def count_outliers(series):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return ((series < lower) | (series > upper)).sum()

    
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(20, rows_plot * 4))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_cols):
        outliers = count_outliers(df[feature].dropna())
        
        sns.boxplot(x=df[feature], ax=axes[i], color=colors[i % len(colors)])
        if outliers > 0:
            axes[i].set_title(f"{feature} → {outliers} Outliers", fontsize=10, color="red")
        else:
            axes[i].set_title(f"{feature} → No Outliers", fontsize=10, color="green")
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_visualizations(
    df, plot_type, x=None, y=None, hue=None, cols_to_plot=None, 
    grid_cols=3, theme='darkgrid', palette='viridis', 
    title_color='#ff7300', title_fontsize=16, **kwargs
):
    plot_mapping = {
        'hist': sns.histplot, 'box': sns.boxplot, 'count': sns.countplot,
        'scatter': sns.scatterplot, 'bar': sns.barplot
    }

    with sns.axes_style(theme if theme != 'custom_dark' else 'darkgrid'):
        if plot_type == 'heatmap':
            print("--- Correlation Heatmap ---")
            plt.figure(figsize=kwargs.pop('figsize', (12, 10)))
            if theme == 'custom_dark':
                plt.gca().set_facecolor('#1E1E1E')
                plt.gcf().set_facecolor('#1E1E1E')
            numeric_df = df.select_dtypes(include=np.number)
            sns.heatmap(numeric_df.corr(), cmap=palette, **kwargs)
            plt.title('Correlation Heatmap', fontsize=title_fontsize, color=title_color, weight='bold')
            plt.xticks(rotation=45, color='white' if theme == 'custom_dark' else 'black')
            plt.yticks(rotation=0, color='white' if theme == 'custom_dark' else 'black')
            plt.show()
            return
            
        if plot_type == 'pairplot':
            print("--- Pairplot ---")
            g = sns.pairplot(df, vars=cols_to_plot, hue=hue, palette=palette, **kwargs)
            g.fig.suptitle('Pairplot of Features', y=1.02, fontsize=title_fontsize, color=title_color, weight='bold')
            if theme == 'custom_dark':
                g.fig.set_facecolor('#1E1E1E')
                for ax in g.axes.flatten():
                    ax.set_facecolor('#1E1E1E')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.tick_params(colors='white', which='both')
            plt.show()
            return
            
        if plot_type not in plot_mapping:
            raise ValueError(f"Invalid plot_type: '{plot_type}'.")
        plot_func = plot_mapping[plot_type]

        if cols_to_plot:
            num_plots = len(cols_to_plot)
            rows = (num_plots + grid_cols - 1) // grid_cols
            fig, axes = plt.subplots(rows, grid_cols, figsize=(grid_cols * 5.5, rows * 4.5))
            axes = axes.flatten()
            if theme == 'custom_dark': fig.set_facecolor('#1E1E1E')

            print(f"--- Grid of {plot_type.capitalize()} Plots ---")
            colors = sns.color_palette(palette, num_plots)
            for i, col in enumerate(cols_to_plot):
                ax = axes[i]
                if theme == 'custom_dark': 
                    ax.set_facecolor('#252525')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')

                if plot_type in ['count', 'bar']:
                    plot_func(x=col, data=df, ax=ax, palette=palette, **kwargs)
                else:
                    plot_func(x=df[col], ax=ax, color=colors[i], **kwargs)
                
                ax.set_title(f'{plot_type.capitalize()} of {col}', fontsize=title_fontsize - 2, color=title_color, weight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')

            for j in range(num_plots, len(axes)): fig.delaxes(axes[j])
            plt.tight_layout(pad=2.0)
            plt.show()

        elif x:
            plt.figure(figsize=kwargs.pop('figsize', (12, 7)))
            ax = plt.gca()
            if theme == 'custom_dark':
                ax.set_facecolor('#252525')
                plt.gcf().set_facecolor('#1E1E1E')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')

            plot_args = {'data': df, 'x': x, 'ax': ax, 'palette': palette}
            if y: plot_args['y'] = y
            if hue: plot_args['hue'] = hue
            plot_func(**plot_args, **kwargs)

            title = f'Count of {x}' if plot_type == 'count' else f'{plot_type.capitalize()} of {x}'
            if y: title = f'{plot_type.capitalize()} of {x} vs. {y}'
            if hue: title += f' (by {hue})'
            ax.set_title(title, fontsize=title_fontsize, color=title_color, weight='bold')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("Please provide 'cols_to_plot' for a grid or 'x' for a specific plot.")


def encode_features(df, cols_to_encode, encoding_type, drop_first=False, **kwargs):
    df_encoded = df.copy()
    fitter = None
    
    print(f"--- Applying {encoding_type.capitalize()} Encoding on {cols_to_encode} ---")

    if encoding_type == 'binary_map':
        fitter = {}
        for col in cols_to_encode:
            unique_vals = df_encoded[col].unique()
            if len(unique_vals) != 2:
                raise ValueError(f"Binary mapping requires exactly 2 unique values, but column '{col}' has {len(unique_vals)}. Consider using 'custom_map' instead.")
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df_encoded[col] = df_encoded[col].map(mapping)
            fitter[col] = mapping

    elif encoding_type == 'custom_map':
        custom_mapping = kwargs.get('custom_mapping')
        if not custom_mapping:
            raise ValueError("For 'custom_map' encoding, please provide the 'custom_mapping' dictionary.")
        
        fitter = {}
        for col in cols_to_encode:
            if col not in custom_mapping:
                raise ValueError(f"Mapping not provided for column '{col}' in the 'custom_mapping' dictionary.")
            
            mapping = custom_mapping[col]
            original_nan_count = df_encoded[col].isnull().sum()
            df_encoded[col] = df_encoded[col].map(mapping)
            fitter[col] = mapping
            
            if df_encoded[col].isnull().sum() > original_nan_count:
                print(f"Warning: New NaN values were introduced in column '{col}'. This may be because some categories in the data were not in your mapping dictionary.")

    elif encoding_type == 'ordinal':
        categories = kwargs.get('categories')
        if not categories:
            raise ValueError("For ordinal encoding, please provide the 'categories' dictionary.")
        
        ordered_cats_list = []
        for col in cols_to_encode:
            if col not in categories:
                raise ValueError(f"Category order not provided for column '{col}'.")
            ordered_cats_list.append(categories[col])
            
        fitter = OrdinalEncoder(categories=ordered_cats_list)
        df_encoded[cols_to_encode] = fitter.fit_transform(df_encoded[cols_to_encode])
            
    elif encoding_type == 'onehot':
        fitter = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first' if drop_first else None)
        encoded_data = fitter.fit_transform(df_encoded[cols_to_encode])
        new_cols = fitter.get_feature_names_out(cols_to_encode)
        encoded_df = pd.DataFrame(encoded_data, index=df_encoded.index, columns=new_cols)
        df_encoded = pd.concat([df_encoded.drop(cols_to_encode, axis=1), encoded_df], axis=1)

    elif encoding_type == 'target':
        target_col = kwargs.get('target_col')
        if not target_col:
            raise ValueError("For target encoding, please provide the 'target_col' name.")
        
        fitter = {}
        global_mean = df_encoded[target_col].mean()
        
        for col in cols_to_encode:
            mapping = df_encoded.groupby(col)[target_col].mean()
            df_encoded[col] = df_encoded[col].map(mapping)
            fitter[col] = {'mapping': mapping, 'global_mean': global_mean}
            
    elif encoding_type == 'frequency':
        fitter = {}
        for col in cols_to_encode:
            freq_map = df_encoded[col].value_counts(normalize=True)
            df_encoded[col] = df_encoded[col].map(freq_map)
            fitter[col] = freq_map

    else:
        raise ValueError(f"Unknown encoding_type: '{encoding_type}'. Choose from 'binary_map', 'custom_map', 'ordinal', 'onehot', 'target', 'frequency'.")

    print("Encoding complete. Displaying the head of the transformed data:")
    display(df_encoded.head())
    
    return df_encoded, fitter

def remove_outliers(df, numeric_cols=None, method='iqr', z_thresh=3, drop=True):

    df_out = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            if drop:
                df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
            else:
                df_out[f"{col}_outlier"] = ~df_out[col].between(lower, upper)
    elif method == 'zscore':
        from scipy.stats import zscore
        for col in numeric_cols:
            z = zscore(df_out[col].fillna(df_out[col].mean()))
            if drop:
                df_out = df_out[(z >= -z_thresh) & (z <= z_thresh)]
            else:
                df_out[f"{col}_outlier"] = abs(z) > z_thresh
    else:
        raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")
    
    print(f"--- Outlier Removal Applied using {method} method ---")
    display(df_out.head())
    
    return df_out


def scale_features(df, numeric_cols=None, method='standard'):
    df_scaled = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'normalize':
        scaler = Normalizer()
    else:
        raise ValueError("Invalid method. Choose from 'standard','minmax','robust','normalize'.")
    
    if method == 'normalize':  
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    else:
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    
    print(f"--- Scaling Method Applied: {method} ---")
    display(df_scaled.head())
    
    return df_scaled, scaler

def feature_engineering(df, datetime_cols=None, binning_dict=None, interaction_pairs=None, drop_original=False):

    df_fe = df.copy()
    
    if datetime_cols:
        for col in datetime_cols:
            df_fe[col] = pd.to_datetime(df_fe[col], errors='coerce')
            df_fe[f"{col}_year"] = df_fe[col].dt.year
            df_fe[f"{col}_month"] = df_fe[col].dt.month
            df_fe[f"{col}_day"] = df_fe[col].dt.day
            df_fe[f"{col}_weekday"] = df_fe[col].dt.weekday
            df_fe[f"{col}_hour"] = df_fe[col].dt.hour
    
    if binning_dict:
        for col, params in binning_dict.items():
            df_fe[f"{col}_binned"] = pd.cut(df_fe[col], bins=params['bins'], labels=params.get('labels', None))
    
    if interaction_pairs:
        for col1, col2 in interaction_pairs:
            df_fe[f"{col1}_x_{col2}"] = df_fe[col1] * df_fe[col2]
    
    if drop_original:
        if datetime_cols: df_fe.drop(columns=datetime_cols, inplace=True)
        if binning_dict: df_fe.drop(columns=list(binning_dict.keys()), inplace=True)
    
    print("--- Feature Engineering Applied ---")
    display(df_fe.head())
    
    return df_fe
def correlation_feature_selection(df, threshold=0.85, method='pearson', caption="Correlation & Feature Selection"):

    corr = df.corr(method=method)
    
    
    high_corr = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                high_corr.add(corr.columns[i])
    
    styled_corr = (
        corr.style
        .background_gradient(cmap='coolwarm')
        .set_caption(caption)
        .set_properties(**{'color':'black','border':'1px solid #444'})
    )
    
    print(f"Highly correlated features (threshold={threshold}): {list(high_corr)}")
    display(styled_corr)
    
    return list(high_corr)


