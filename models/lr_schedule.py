def adjust_learning_rate(optimizer, base_lr, iters, max_iters, power = 0.9):
    lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
