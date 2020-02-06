import torch

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