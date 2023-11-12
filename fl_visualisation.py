# some helper functions for plotting images, visualising loss curves, etc.

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL.Image import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, rcParams
import time
import asyncio

from IPython.display import display, display_pretty, clear_output


#### functions to display images from datasets:

def disp_images(*imgs: np.ndarray, 
         pred_labels: list=None, 
         true_labels: list=None,
         h_pad:        int=5,
         imgs_per_row: int=5):
    """accepts any number of images, and displays them on a figure.

    accepts an optional list as kwarg 'pred_labels', which indicates
        the predicted labels of the images provided.
        if supplied, will write those labels on the output.
    
    also accepts another optional list, 'true_labels', which
        will allow the output to indicate where the predicted
        labels are incorrect.
    """
    
    # quietly re-parse single list passed as first arg:
    if len(imgs) == 1 and isinstance(imgs[0], list):
        imgs = tuple(imgs[0])
    # and/or a stacked tensor
    if isinstance(imgs, torch.Tensor) and len(imgs.shape) > 2:
        imgs = [img for img in imgs]

    # cast to numpy arrays if not already:
    arrs = [np.asarray(img) for img in imgs]
    img_dims = arrs[0].shape

    num_imgs = len(imgs)
    num_rows = int(np.ceil(num_imgs / imgs_per_row))
    fig, axes = plt.subplots(num_rows, imgs_per_row)

    for i, arr in enumerate(arrs):           
        # clip to interval [0,1]:
        output = arr.clip(0,1)

        # plot image as subplot:
        row_num, col_num = divmod(i, imgs_per_row)
        if num_rows == 1:
            ax = axes[col_num] if num_imgs > 1 else axes
        else:
            ax = axes[row_num][col_num]
        ax.imshow(output, cmap='Blues', vmin=0, vmax=1)

        # hide tick labels:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # but add class labels if given:
        if (true_labels is not None) and (pred_labels is None):
            # just plot the true labels
            pred_labels = true_labels
            true_labels = None
        
        if pred_labels is not None:
            if true_labels is not None:
                # display predicted labels in black if correct,
                # or both predicted/true labels in red if incorrect
                correct = pred_labels[i] == true_labels[i]
                title_colour = 'black' if correct else 'red'
                title = f'✔ {pred_labels[i]}' if correct else f'✗ {pred_labels[i]}\n✔ {true_labels[i]}'
            else:
                title_colour='black'
                title = pred_labels[i]
            ax.set_title(title, c=title_colour)

    fig.tight_layout(h_pad=h_pad)
    plt.show()

def inspect_data(dataset: Dataset,
                 num_images: int=5):
    """accepts a torch Dataset as first arg, and randomly displays
    a few images from it in a row.
    
    if class labels are given, display those as well."""

    num_datapoints = len(dataset)
    num_classes = len(dataset.classes)
    
    random_idxs = np.random.permutation(num_datapoints)[:num_images]

    datapoints = [dataset[i] for i in random_idxs]
    imgs = [dp[0] for dp in datapoints]
    labels = [dp[1] for dp in datapoints]
    label_names = [dataset.classes[l] for l in labels]
    disp_images(imgs, true_labels=label_names, h_pad=-5)

def inspect_model_outputs(model: nn.Module,
                          dataset: Dataset,
                          num_images: int = 15):
    """accepts a pytorch model object and an image dataset;
    generates some data batches from the dataset and compares
    model outputs to ground truth labels."""
    data_iter = iter(DataLoader(dataset, batch_size=num_images, shuffle=True))
    batch_x, batch_y = next(data_iter)
    with torch.no_grad():
        batch_out = model(batch_x)
    batch_labels = torch.argmax(batch_out, 1)
    images = [np.asarray(img) for img in batch_x]

    batch_classnames = [dataset.classes[y] for y in batch_labels]
    true_classnames = [dataset.classes[y] for y in batch_y]
    disp_images(images, pred_labels=batch_classnames, true_labels=true_classnames, h_pad=-3)



#### graph plotting functions:

def plot_label_distribution(partitions: list[Dataset],
                            show_expected=True):
    """plots the distributionn of class labels within a collection
    of data Subsets, as a clustered bar chart."""
    
    num_partitions = len(partitions)
    total_samples = sum([len(part) for part in partitions])

    # assume all datasets share the same class mapping:
    class_names = partitions[0].classes
    num_classes = len(class_names)

    x_loc = np.arange(num_partitions)
    cluster_width = num_classes * 1.5
    bar_width = 1/cluster_width

    fig, ax = plt.subplots(layout='constrained')
    
    samples_per_class_per_part = []

    cmap = colormaps['Blues']

    for p, partition in enumerate(partitions):
        labels = partition.labels
        label_idxs, label_counts = np.unique(labels, return_counts=True)
        # account for 0 labels in some datasets:
        full_counts = {i: 0 for i in range(num_classes)}
        for idx, count in zip(label_idxs, label_counts):
            full_counts[idx] = count
        assert list(label_idxs) == sorted(label_idxs)
        samples_per_class_per_part.append(full_counts)

    if show_expected:
        # plot the expected bar heights given homogeneous stratification:
        # total_samples = sum([sum([v for v in part.values()]) for part in samples_per_class_per_part])
        exp_count = (total_samples / num_partitions) / num_classes
        ax.axhline(exp_count, linestyle=':', c=cmap(0.5), label="'Expected' equal distribution", zorder=1)

    
    part_labels = np.arange(num_partitions)
    for c, name in enumerate(class_names):
        # plot the cluster on figure axis:
        offset = bar_width * c - (3.5 / cluster_width)
        counts_per_part = [part[c] for part in samples_per_class_per_part]
        # pick a color by drawing from the colormap between range 128-255
        color = cmap((c / (num_classes*2)) + 0.5)
        rects = ax.bar(x_loc + offset, counts_per_part, bar_width, fc=color, zorder=2, label=None)
        # ax.bar_label(rects, padding=3)

    # indicate missing classes:
    miss_xs = []
    for p in range(num_partitions):
        for c in range(num_classes):
            if samples_per_class_per_part[p][c] == 0:
                bar_loc = x_loc[p] + (bar_width * c - (3.5 / cluster_width))
                miss_xs.append(bar_loc)
    if len(miss_xs) > 0:
        marker_height = total_samples / num_partitions / num_classes / 50
        plt.plot(miss_xs, [marker_height]*len(miss_xs), 'r^', fillstyle='none', label='Missing classes')
    
    ax.set_xlabel('Partition')
    ax.set_ylabel('# of samples in class')
    ax.set_title('Class distribution by partition')

    # minor ticks are the centres of each cluster:
    ax.set_xticks(x_loc + bar_width, np.arange(num_partitions), minor=True)
    # and major ticks are the boundary between them:
    ax.set_xticks(x_loc + bar_width + 0.5, [], minor=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='x', which='minor', length=0)

    
    ax.legend()# loc='upper right', ncols=3)

    plt.show()

    # return samples_per_class_per_part


def plot_series(*series: list[float],
              rolling_window: int = 500, # size of rolling window for curve smoothing
              epochs: int = None, # optional number of epochs to demarcate plot):
              epoch_learning_rates: dict = None, # optional mapping of epoch numbers to LR values
              names=None, styles=None,
              xlabel=None, ylabel=None, title=None, 
              ax=None, show=True,
              live=False, cur_step=None, max_step=None,
              ):
    """accepts an iterable of lists of floats interpreted as some training metric per update step,
    and plots them both on the specified axis (or a new axis if not given).
        other optional args:
    - 'names' and 'styles': iterables of strings denoting name and style for each series.
    - 'rolling_window', smooths the resulting curve with a box filter of that size.
    - 'epochs', if provided, adds tick marks to the plot denoting epoch boundaries."""

    if ax is None:
        ax = plt.gca()

    if len(series) > 10:
        # assume we've been fed a single array of values by mistake,
        # which we quietly wrap into a list before continuing:
        series = [series]
    num_steps = len(series[0])
    assert len(series[0]) == len(series[-1]), "all input series for plotting must be of equal length"
    steps = np.arange(num_steps)

    for s, values in enumerate(series):
        name = names[s] if names is not None else None
        style = styles[s] if styles is not None else None

        # plot raw values as dotted line:
        if (rolling_window is not None) and (rolling_window > 1):
            # smooth the values with a rolling average before plotting:
            # then smooth that line using rolling average:
            smooth_values = np.convolve(values, np.ones(rolling_window), 'valid') / rolling_window
            # but pad from the start so it centres on the correct point:
            smooth_start_idx = rolling_window // 2
            smooth_values = [None]*smooth_start_idx + list(smooth_values)
            smooth_steps = np.arange(len(smooth_values))
            if live:
                ax.set_data(smooth_steps, smooth_values)
            else:
                ax.plot(smooth_steps, smooth_values, style, label=name, zorder=-s)
        else:
            # just plot raw values
            plt.plot(steps, values, style, label=name, zorder=-s)

    # these only need to be called for initial drawing:
    if (not live) or (cur_step > 0):
        if max_step is not None:
            ax.set_xlim([0, max_step])
        else:
            max_step = num_steps
        if xlabel is None:
            if epochs is not None:
                steps_per_epoch = max_step // epochs
                epoch_marks = np.arange(epochs) * steps_per_epoch
                ax.set_xticks(epoch_marks, minor=False)
                ax.set_xticklabels(np.arange(epochs)+1, minor=False)
                ax.set_xlabel('Epoch')
                if epoch_learning_rates is not None:
                    # use minor tick points to record learning rates
                    # but only if there's enough space on the graph:
                    if epochs <= 3:
                        epoch_centre_marks = epoch_marks + (steps_per_epoch / 2)
                        ax.set_xticks(epoch_centre_marks, minor=True)
                        lr_strings = [f'lr={epoch_learning_rates[e]:.0e}' for e in range(epochs)]
                        ax.set_xticklabels(lr_strings, minor=True)
                        ax.tick_params(axis='x', which='minor', length=0, labelsize=7)
            else:
                ax.set_xlabel('Training step')
        else:
            ax.set_xlabel(xlabel)
        
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    if show:
        plt.show()


def plot_metrics(train_loss, val_loss, train_acc, val_acc, rolling_window=500,
                 epochs=None, lrs=None,
                 fig=None, **kwargs):
    """accepts 4 lists of floats interpreted as losses/accuracies per update step.
    optional args:
    - 'rolling_window', smooths the resulting curve with a box filter of that size.
    - 'epochs', if provided, adds tick marks to the plot denoting epoch boundaries."""

    # get current figure size params:
    fig_width, fig_height = rcParams['figure.figsize']
    
    # get the figure, retrieving one if one already exists:
    if fig is None:
        # set height to half of width for drawing two plots:
        rcParams['figure.figsize'] = fig_width, fig_width/2
        fig, axes = plt.subplots(1, 2)
    else:
        axes = fig.axes

    if epochs is not None and lrs is not None:
        epoch_learning_rates = []
        num_epochs = max(epochs)
        for e in range(num_epochs):
            epoch_lr = lrs[np.where(np.asarray(epochs)==e)[0][0]]
            epoch_learning_rates.append(epoch_lr)

    # two plots: loss on the left, accuracy on the right
    loss_ax, acc_ax = axes
    plot_series(train_loss, val_loss, 
                rolling_window=rolling_window,
                names=['Training', 'Validation'],
                styles=[':r', ':b'],
                title='Loss', ylabel=f'Loss (rolling average, k={rolling_window})',
                ax=loss_ax, show=False,
                epoch_learning_rates=epoch_learning_rates,
                **kwargs)
    plot_series(train_acc, val_acc, 
                rolling_window=rolling_window,
                names=['Training', 'Validation'],
                styles=['-r', '-b'],
                title='Accuracy', ylabel=f'Accuracy (rolling average, k={rolling_window})',
                ax=acc_ax, show=False, 
                epoch_learning_rates=epoch_learning_rates,
                **kwargs)
    plt.tight_layout()
    plt.show()

    # restore existing figure size params:
    rcParams['figure.figsize'] = fig_width, fig_height





