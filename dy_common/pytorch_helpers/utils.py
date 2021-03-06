import random
import torch


def get_cuda_memory_usage():
    """
    Returns current memory usage as a list as a list across all devices, and max memory usage as a list across all
    devices
    """
    bytes_to_gb = 1e-9
    return [torch.cuda.memory_allocated(device=idx) * bytes_to_gb for idx in range(torch.cuda.device_count())], \
           [torch.cuda.max_memory_allocated(device=idx) * bytes_to_gb for idx in range(torch.cuda.device_count())]


def dataset_resample(data_dict, class_list, sample_distribution):
    """
    Takes an input dict (data_dict), with dictionary values being the class labels, and returns a list of (key, value)
    pairs which are randomly sampled according to sample_distribution, among the class samples. This is specifically
    for image datasets.

    Assumes data_dict is a dictionary of the following format: {'path_to_imagefile': class_string_or_int}, and assumes
    len(sample_distibution) == len(class_list) == number of classes in data_dict, and in the same order as class_list

    If class_list = None, assume that class_list = range(len(sample_distribution))

    If sample_distribution is negative for a class, then those samples are not resampled, and the entire list is used

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
        if sample_distribution[i] < 0:
            resampled_class_lists[i] = class_lists[i].copy()
        elif sample_distribution[i] > len(class_lists[i]):
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


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Warmup LR scheduler, from https://github.com/pytorch/vision/blob/master/references/detection/utils.py

    This returns a LambdaLR, which multiplies the original LR by a factor given here by f(epoch). Since, in our usage,
    the epoch increments when we call lr_scheduler.step(), this LR should iterate every batch of the first epoch instead
    of every epoch.

    With the usage below, LR will go from ~0 to the lr of the optimizer, in either 1000 increments, or len(data_loader)
    increments, depending on which is smaller.

    Usage:
    train_one_epoch(..., data_loader, epoch, ...):
        ...
        lr_scheduler = None
        if epoch == 0:
            print('Starting warmup epoch...')
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        ...
        for batch in data_loader:
            ...
            if lr_scheduler is not None:
                lr_scheduler.step()
            ...
    """

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def load_state_dict_lenient(model, state_dict):
    """
    A function which tries to load state_dict into model. If it fails, then try to load any state dict entries which
    have the same name and shape as the model's state dict.
    """

    try:
        model.load_state_dict(state_dict)
    except:
        print('Warning: Name and/or shape mismatch when loading state dict. Attempting to load only matching entries.')
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if
                               (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        load_state_dict_result = model.load_state_dict(filtered_state_dict, strict=False)

        print('Loaded state dict. Missing keys: {}, unexpected keys: {}'.format(load_state_dict_result.missing_keys,
                                                                                load_state_dict_result.unexpected_keys))


if __name__ == '__main__':
    p = torch.tensor([1])
    optimizer = torch.optim.SGD(params=[p], lr=1)

    # a test to see how warmup_lr_scheduler works
    data_loader_len = 500
    warmup_factor = 1./1000
    warmup_iters = min(1000, data_loader_len - 1)

    lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i in range(data_loader_len):
        lr_scheduler.step()
        print(optimizer.param_groups[0]['lr'])
