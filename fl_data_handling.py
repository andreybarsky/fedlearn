import numpy as np
from dataclasses import dataclass
import torch
from torch import utils, Tensor
from torch.utils.data import Dataset, TensorDataset, Subset
from torchvision import transforms

# class PartitionData(utils.data.Dataset):
#     """custom Dataset subclass meant to represent private partition data,
#     stored in memory instead of on disk for simplicity"""
#     def __init__(self, data, targets):
#         self.data = data
#         self.targets = torch.LongTensor(targets)
#         self.transform = transform
        
#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.targets[index]
        
#         if self.transform:
#             x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
#             x = self.transform(x)
        
#         return x, y
    
#     def __len__(self):
#         return len(self.data)

def load_data(dataset_class: type, # a callable torch dataset class
              val_ratio: float=0.1):
    """
    accepts a callable torch dataset class,
    downloads the corresponding data if necessary,
    and loads into memory.

    Dataset objects are fetched for the remote train and test sets,
    and the original train set is further divided into train and validation.

    also accepts a float 'val_ratio' between 0-1, interpreted as the ratio
    of the training data to set aside as a validation set.

    returns custom Dataset objects with .x, .y and .classes attributes."""


        
    data_main = dataset_class('./data/train', 
                              train=True,
                              download=True)
    
    data_test = dataset_class('./data/test', 
                              train=False,
                              download=True)

    
    # convert these to arrays for now so we can work with them more easily:
    x_main, y_main = data_main.data.numpy(), data_main.targets.numpy()
    x_test, y_test = data_test.data.numpy(), data_test.targets.numpy()

    if not isinstance(x_main.dtype, float):
        # rescale integer 0-255 pixel values to 0-1 floats:
        x_main = x_main.astype(float) / 255.
        x_test = x_test.astype(float) / 255.
    
    # split main set into training and validation sets:
    num_val = int(len(x_main) * val_ratio)
    val_idxs = np.random.permutation(len(x_main))[:num_val]
    # using index by boolean mask:
    val_idx_mask = np.zeros(len(x_main), dtype=bool)
    val_idx_mask[val_idxs] = 1
    x_val, y_val = x_main[val_idx_mask], y_main[val_idx_mask]
    x_train, y_train = x_main[~val_idx_mask], y_main[~val_idx_mask]

    # package them up as TensorDatasets:
    data_train = TensorDataset(Tensor(x_train), Tensor(y_train).long())
    data_val = TensorDataset(Tensor(x_train), Tensor(y_train).long())
    data_test = TensorDataset(Tensor(x_train), Tensor(y_train).long())

    for obj in data_train, data_val, data_test:
        # assign neat attributes for x and y tensors:
        obj.x = obj.data = obj.tensors[0]
        obj.y = obj.targets = obj.labels = obj.tensors[1]
        # pass class name mappings too:
        obj.classes = data_main.classes

    # return all three sets:
    return data_train, data_val, data_test



def partition_data(dataset: Dataset,
                   num_partitions: int,
                       # i.e. the number of client machines which will 
                       #     contribute federated learning updates.
                   heterogeneity: str='strong',
                       # one of 'none', 'weak', or 'strong'.
                   device='cpu',
                  ):
    """accepts a Dataset object, and divides it into num_partitions
    subsets, each representing a fraction of the data held privately 
    by some client.
    
    arg 'heterogeneity' is one of: 'none', 'weak', or 'strong':
        - 'none' is perfect homogeneity: all partitions 
            contain all class labels equally.
        - 'weak' randomly skews the distribution of class 
            labels across partitions.
        - 'strong' has random skew and additionally creates 
            partitions that lack some class labels entirely.
    
    returns num_partitions new Dataset objects as a list."""
    
    num_classes = len(dataset.classes)
    total_samples = len(dataset)
    
    # handle strong heterogeneity, and work out
    # which classes are dropped in which partitions:
    if heterogeneity == 'strong':
        assert num_classes > 2, "cannot have missing classes for binary classification"
        
        if num_classes >= num_partitions:
            # if at least as many classes as partitions,
            # drop 1/K of classes from each of the K partitions
            fractions_to_drop = np.linspace(0, 1, num_partitions+1)
            idxs = (fractions_to_drop * num_classes).astype(int)
            
            # randomise which to drop so the ordering isn't sequential:
            shuffled_classes = list(np.random.permutation(range(num_classes)))
            classes_to_drop = [shuffled_classes[idxs[i] : idxs[i+1]]
                               for i in range(num_partitions)]
        
        elif num_partitions > num_classes:
            # or if more partitions than classes,
            # each partition will just drop 1 class.
            
            # first, each class is dropped at least once:
            class_to_drop = list(range(num_classes))
            
            # then sample from classes uniformly for each remaining partition:
            for i in range(num_partitions - num_classes):
                class_to_drop.append(np.random.choice(num_classes))
            # but format as list of lists for each partition, 
            # to match the classes >= partitions case above
            classes_to_drop = [[c] for c in class_to_drop]
        
        # finally, make the list that describes which classes each partition
        # DOES have access to, i.e. which ones are not dropped
        partition_classes = [[c for c in range(num_classes) if c not in classes_to_drop[p]] for p in range(num_partitions)]
        # and get the reverse mapping: which class is represented in which partitions
        class_partitions = [[p for p in range(num_partitions) if c in partition_classes[p]] for c in range (num_classes)] 
    
    else:
        # all partitions contain all classes:
        partition_classes = [list(range(num_classes)) for p in range(num_partitions)]
        class_partitions = [list(range(num_partitions)) for c in range(num_classes)]
        
    # now, handle 'weak' homogeneity (which is also included in strong homogeneity).
    # i.e. skewed distribution of classes between partitions which contain that class.
    
    # we'll populate this dict that enumerates the datapoint indices for each partition:
    partition_data_idxs = {p: [] for p in range(num_partitions)}

    samples_allocated = 0
    
    
    for c in range(num_classes):
        # print(f'\nLooping across class {c}')
        # get the indices in the dataset where this class occurs:
        class_example_idxs = np.where(dataset.targets == c)[0]
        num_class_datapoints = len(class_example_idxs)
        # print(f'{num_class_datapoints} samples of this class to allocate')
        # get the partitions that include this class:
        relevant_partitions = class_partitions[c]
        num_relevant = len(relevant_partitions)
        # print(f'  Where {num_relevant} partitions are relevant: {relevant_partitions}')
        
        # under perfect homoegeneity, we stratify the split such that each
        # relevant partition contains an equal fraction of a class's examples,
        # represented as equal spacing along the [0,1] number line:
        equal_distribution = np.linspace(0,1,num_relevant+1)
        # print(f'{equal_distribution =}')
        
        # but if heterogeneity is at least weak, we perturb this distribution
        # to skew the distribution of classes across partitions:
        if heterogeneity in ('weak', 'strong'):
            # (I tried sampling uniformly across the number line for this,
            # but it leads to too many cases where boundaries are clustered together
            # resulting in almost no samples of a class, so instead this is an
            # iterative process with some guaranteed minimum number of datapoints per
            # class, equal to 20% of the expected equal allocation:
            tol = equal_distribution[1] * 0.2
            # print(f'{tol=}')
            skewed_distribution = [equal_distribution[0]]
            for r in range(1,num_relevant):
                # sample uniformly from between the previous 'skewed' boundary
                # and the next 'equal' one:
                l_bound = skewed_distribution[-1] + tol
                u_bound = equal_distribution[r+1] - tol
                # print(f'  partition {relevant_partitions[r]} will end between bounds: {l_bound:.2f},{u_bound:.2f}')
                bound = np.random.uniform(l_bound, u_bound)
                skewed_distribution.append(bound)
            # as first bound is 0, final bound is 1:
            # print(f'final partition ends at 1')
            skewed_distribution.append(1)
            # print(f'{skewed_distribution = }')
            
            # use these perturbed points along the number line as a discrete PD:
            class_distribution = np.asarray(skewed_distribution)
        else:
            # non-heterogeneous, stratified split:
            class_distribution = equal_distribution
        
        # now translate those distributions to integer indices into the dataset 
        boundaries = (class_distribution * num_class_datapoints).astype(int)
        shuffled_class_example_idxs = np.random.permutation(class_example_idxs)
        data_idxs_per_rpartition = [shuffled_class_example_idxs[boundaries[r] : boundaries[r+1]] 
                                   for r in range(num_relevant)]
        num_samples_per_rpartition = [len(idxs) for idxs in data_idxs_per_rpartition]
        
        # print(f'Allocated class {c} samples for {num_relevant} partitions: {num_samples_per_rpartition}')
        samples_allocated += sum(num_samples_per_rpartition)
        frac_allocated = samples_allocated / total_samples
        # print(f'  {samples_allocated}/{total_samples} samples allocated ({frac_allocated:.0%})')
        
        for r, idxs in enumerate(data_idxs_per_rpartition):
            # and, since we may be working with only a subset of all partitions
            # (because not all are guaranteed to contain this class)
            # we have to allocate them back into the appropriate partitions:
            p = relevant_partitions[r]
            partition_data_idxs[p].extend(data_idxs_per_rpartition[r])
        # this process is repeated for each class.
        
    # finally, create Dataset objects from the original Dataset 
    # which correspond to this partitioning scheme we've created

    partition_datasets = []
    for p in range(num_partitions):
        idxs = partition_data_idxs[p]
        part_x, part_y = dataset[idxs]
        part_data = TensorDataset(part_x, part_y)
        # assign accessor attributes:
        part_data.x = part_data.data = part_data.tensors[0]
        part_data.y = part_data.labels = part_data.targets = part_data.tensors[1]
        part_data.classes = dataset.classes
        
        partition_datasets.append(part_data)

    # verify that all samples are accounted for:
    total_partitioned_samples = sum([len(part) for part in partition_datasets])
    assert total_partitioned_samples == len(dataset), f"Missing samples during partitioning process - only {total_partitioned_samples}/{len(dataset)} accounted for ({(total_partitioned_samples / len(dataset)):.1%})"
    
    return partition_datasets

def num_correct(pred: Tensor,
                true: Tensor):
    """accepts two Tensors:
    pred: model output of the shape (batch, num_classes)
    true: dataset labels of the shape (batch,) with values as integer class indices
    and returns the number of correct classifications
    by taking argmax of the prediction logits."""
    with torch.no_grad():
        pred_indices = torch.argmax(pred, 1)
        correct = sum(true == pred_indices)
    return correct

def get_val_metrics(net, val_loader, val_iter, loss_func):
    with torch.no_grad():
        try:
            val_x, val_y = next(val_iter)
        except StopIteration:
            # if we've hit the end of the validation data,
            # just loop back around again
            val_iter = iter(val_loader)
            val_x, val_y = next(val_iter)
        val_out = net(val_x)
    val_loss = loss_func(val_out, val_y).item()
    val_acc = num_correct(val_out, val_y) / len(val_out)
    return val_loss, val_acc, val_iter

def get_class_distribution(dataset):
    """accepts a torch Dataset object and returns a dict
    that keys class label ID to the proportion of samples
    in the dataset with that class"""
    if isinstance(dataset, Subset):
        # get the labels by indexing the original dataset
        # with this horrible namespace:
        labels = dataset.dataset.labels[dataset.indices]
    else:
        labels = dataset.labels
    total_samples = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    proportions = {cls: count/total_samples for cls,count in zip(classes, counts)}
    return proportions
        