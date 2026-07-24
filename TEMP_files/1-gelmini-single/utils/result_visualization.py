import matplotlib.pyplot as plt


# Visualize top k resources/activities based on its frequency
def visualize_top_k(res_freq, k, mode, title, x_ticks_rotation=45):
    if mode == "resources":
        label = "Resources"
    elif mode == "actions":
        label = "Actions"
    else:
        label = "Activities"
    top_k_resources = dict(sorted(res_freq.items(), key=lambda x: x[1], reverse=True)[:k])
    plt.figure(figsize=(20, 6))
    bars = plt.bar(top_k_resources.keys(), top_k_resources.values())
    plt.xlabel(label)
    plt.ylabel('Frequency')
    # plt.title(f'Top {k} {label} by frequency using {method} approach')
    plt.title(title)
    plt.xticks(rotation=x_ticks_rotation)
    
    # Add frequency labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
    
    plt.show()
    
def compare_top_k(top_k_1st_method, top_k_2nd_method, method_label_1, method_label_2, mode):
    fig, ax = plt.subplots(figsize=(20, 6))

    # Define x positions for the bars
    x = range(len(top_k_1st_method))
    y1 = list(top_k_1st_method.values())
    y2 = [top_k_2nd_method.get(key, 0) for key in top_k_1st_method.keys()]
    labels = list(top_k_1st_method.keys())

    # Plot the bars
    bars1 = ax.bar([i - 0.2 for i in x], y1, width=0.4, label=method_label_1, color='blue')
    bars2 = ax.bar([i + 0.2 for i in x], y2, width=0.4, label=method_label_2, color='orange')

    # Add frequency labels on top of each bar
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha='center')

    ax.set_ylabel('Frequency')
    ax.set_title(f'Comparison of {mode} frequencies of {method_label_1} and {method_label_2} Methods')
    ax.legend()

    plt.tight_layout()
    plt.show()
