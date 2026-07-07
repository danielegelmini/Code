import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from pathlib import Path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize_remaining_time(case_study, train_data, test_data, save_plot=True):
    try:
        import matplotlib
        matplotlib.use('TkAgg') # Or 'Qt5Agg' depending on your installed packages
    except Exception:
        pass # Fallback safely if backend switching fails

    scaler = StandardScaler()

    # Standardize the "remaining time"
    train_data['standardscaller'] = scaler.fit_transform(train_data[['remaining_time']])
    test_data['standardscaller'] = scaler.transform(test_data[['remaining_time']])

    # Apply the sigmoid function to the standardized "remaining time"
    train_data['sigmoid'] = sigmoid(train_data['standardscaller'])
    test_data['sigmoid'] = sigmoid(test_data['standardscaller'])

    # Apply Min-Max scaling to the sigmoid values
    mm_scaler = MinMaxScaler()  
    train_data['sigmoid_mm'] = mm_scaler.fit_transform(train_data[['sigmoid']])
    test_data['sigmoid_mm'] = mm_scaler.transform(test_data[['sigmoid']])

    # Remove unnecessary columns
    train_data.drop(columns=['standardscaller', 'sigmoid'], inplace=True)
    test_data.drop(columns=['standardscaller', 'sigmoid'], inplace=True)

    # Explicitly create a new, numbered figure window
    plt.figure(num="Normalization Plot", figsize=(8, 5))

    datasets = [
        ("Train", train_data['sigmoid_mm'], "blue"),
        ("Test", test_data['sigmoid_mm'], "orange"),
    ]

    # Compute common bin edges
    all_data = np.concatenate([d[1] for d in datasets])
    min_val, max_val = all_data.min(), all_data.max()
    bins = np.linspace(min_val, max_val, 30)

    # Plot both histograms
    for title, data, color in datasets:
        sns.histplot(
            data, bins=bins, kde=False, stat='density',
            alpha=0.6, color=color, label=title
        )

    # Compute stats
    stats_text = ""
    for title, data, color in datasets:
        min_val = data.min()
        max_val = data.max()
        stats_text += (
            f"{title}:\n"
            f"  Min = {min_val:.2f}\n"
            f"  Max = {max_val:.2f}\n"
        )

    # Add stats box
    anchored_text = AnchoredText(stats_text.strip(), loc="upper right",
                                prop=dict(size=9), frameon=True)
    plt.gca().add_artist(anchored_text)

    # Labels, legend, and layout
    plt.title("Normalization")
    plt.xlabel("Remaining Time (Normalized)")
    plt.ylabel("Density")
    plt.legend(loc=[0.87, 0.62])
    plt.ylim([0, 8.5])
    plt.tight_layout()
    
    output_dir = Path(f"./case_studies/{case_study}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "normalization_plot.png"
    
    if save_plot:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to: {save_path}")
    
    plt.show()

    return train_data, test_data