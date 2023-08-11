import matplotlib.pyplot as plt
import torch

std =  torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()

def viz_maskige(loader, model, i_iter, config):
    for x, y in loader:
        x = x[:3].cuda()
        y = y[:3].cuda()
        break
    with torch.no_grad():
        p, im, imr = model.M.forward_maskige_test(y)
    p = p.argmax(dim=1).float()
    y = y.argmax(dim=1).float()
    print(p.sum())
    fig, ax = plt.subplots(ncols = 5, nrows = 3)
    for i in range(3):
        ax[i, 0].imshow(x[i].cpu().permute(1, 2, 0))
        ax[i, 0].axis('off')

        ax[i, 1].imshow(y[i].cpu(), cmap = 'gray')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(im[i].cpu().permute(1, 2, 0))
        ax[i, 2].axis('off')

        ax[i, 3].imshow(imr[i].cpu().permute(1, 2, 0))
        ax[i, 3].axis('off')

        ax[i, 4].imshow(p[i].cpu(), cmap='gray')
        ax[i, 4].axis('off')

    plt.savefig(f"{config['output_dir']}/viz_post/{i_iter}_maskige.png")


def viz_seg(loader_s, loader_t, model, i_iter, config):
    for x, y in loader_s:
        xs = x[:3].cuda()
        ys = y[:3].cuda()
        break

    for x, y in loader_t:
        xt = x[:3].cuda()
        yt = y[:3].cuda()
        break

    with torch.no_grad():
        fs = model.I(xs)
        ft = model.I(xt)
        ps, ims = model.M(fs)
        pt, imt = model.M(ft)
    ps = ps.argmax(dim=1)
    pt = pt.argmax(dim=1)
    ys = ys.argmax(dim=1)
    yt = yt.argmax(dim=1)

    xs = xs * std + mean
    xt = xt * std + mean

    fig, ax = plt.subplots(ncols = 8, nrows = 3, figsize = (36, 36))
    for i in range(3):
        ax[i, 0].imshow(xs[i].permute(1, 2, 0).cpu())
        ax[i, 0].axis('off')
        ax[i, 1].imshow(ys[i].cpu(), cmap='gray')
        ax[i, 1].axis('off')
        ax[i, 2].imshow(ims[i].permute(1, 2, 0).cpu())
        ax[i, 2].axis('off')
        ax[i, 3].imshow(ps[i].cpu(), cmap='gray')
        ax[i, 3].axis('off')
        ax[i, 4].imshow(xt[i].permute(1, 2, 0).cpu())
        ax[i, 4].axis('off')
        ax[i, 5].imshow(yt[i].cpu(), cmap='gray')
        ax[i, 5].axis('off')
        ax[i, 6].imshow(imt[i].permute(1, 2, 0).cpu())
        ax[i, 6].axis('off')
        ax[i, 7].imshow(pt[i].cpu(), cmap='gray')
        ax[i, 7].axis('off')
    plt.tight_layout()
    plt.savefig(f"{config['output_dir']}/viz_prior/{i_iter}_prior.png")
