import random
import torch

def dataset_resample(data_dict, class_list, sample_distribution):
    """
    Takes an input dict (data_dict), with dictionary values being the class labels, and returns a list of (key, value)
    pairs which are randomly sampled according to sample_distribution, among the class samples. This is specifically
    for image datasets.

    Assumes data_dict is a dictionary of the following format: {'path_to_imagefile': class_string_or_int}, and assumes
    len(sample_distibution) == len(class_list) == number of classes in data_dict, and in the same order as class_list

    If class_list = None, assume that class_list = range(len(sample_distribution))

    For example:
    class_list = ['classa', 'classb', 'classc']
    sample_distribution = (1000, 800, 800)

    or:
    class_list = [0, 1, 2, 3]
    sample_distribution = (100, 100, 100, 100)
    """

    # if None was passed as class_list, assume the labels are just 0 to len(sample_distribution) - 1
    if class_list == None:
        class_list = range(len(sample_distribution))

    num_classes = len(class_list)

    assert len(sample_distribution) == num_classes, \
        'Sample distribution values have different length ({}) than number of classes provided ({})'.format(len(sample_distribution), num_classes)

    # separate data_dict into distinct class lists
    class_lists = [[] for _ in range(num_classes)]
    for key, value in data_dict.items():
        class_lists[class_list.index(value)].append((key, value))

    # resample each class into lists. Sample with replacement if we are sampling more samples than exist, otherwise
    # don't use replacement
    resampled_class_lists = [[] for _ in range(num_classes)]
    for i in range(num_classes):
        if sample_distribution[i] > len(class_lists[i]):
            resampled_class_lists[i] = random.choices(class_lists[i], k=sample_distribution[i])
        else:
            resampled_class_lists[i] = random.sample(class_lists[i], k=sample_distribution[i])

    # join into a single list
    resampled_list = sum(resampled_class_lists, [])

    # shuffle
    random.shuffle(resampled_list)

    return resampled_list

def write_histograms(writer, model, epoch):
    """
    A function to write histograms of all model weights, biases, weight gradients, and bias gradients to a
    torch.utils.tensorboard.SummaryWriter object (writer)
    """

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            print(name)
            if module.weight is not None:
                writer.add_histogram('conv/weights/' + name + '.weight', module.weight, epoch)
                writer.add_histogram('conv/gradients/' + name + '.weight.grad', module.weight.grad, epoch)
            if module.bias is not None:
                writer.add_histogram('conv/biases/' + name + '.bias', module.bias, epoch)
                writer.add_histogram('conv/gradients/' + name + '.bias.grad', module.bias.grad, epoch)
        elif isinstance(module, torch.nn.Conv2d):
            if module.weight is not None:
                writer.add_histogram('conv/weights/' + name + '.weight', module.weight, epoch)
                writer.add_histogram('conv/gradients/' + name + '.weight.grad', module.weight.grad, epoch)
            if module.bias is not None:
                writer.add_histogram('conv/biases/' + name + '.bias', module.bias, epoch)
                writer.add_histogram('conv/gradients/' + name + '.bias.grad', module.bias.grad, epoch)
        elif isinstance(module, torch.nn.Linear):
            if module.weight is not None:
                writer.add_histogram('linear/weights/' + name + '.weight', module.weight, epoch)
                writer.add_histogram('linear/gradients/' + name + '.weight.grad', module.weight.grad, epoch)
            if module.bias is not None:
                writer.add_histogram('linear/biases/' + name + '.bias', module.bias, epoch)
                writer.add_histogram('linear/gradients/' + name + '.bias.grad', module.bias.grad, epoch)