import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def disribution_per_chanel(*mean_of_chanel) -> plt:
    """
    Plot the distribution of the average value per image for each channel.

    Parameters:
    mean_r (float): Mean value for the red channel.
    mean_g (float): Mean value for the green channel.
    mean_b (float): Mean value for the blue channel.

    Returns:
    plt: The matplotlib plot object.
    """
    plt.figure(figsize=(5, 5))
    for mean in mean_of_chanel:
        plt.hist(mean, 20)
    plt.title("Distribution of the average value per image for each channel")
    plt.show()
    return plt