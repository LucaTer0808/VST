import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet
from torch.utils import data
import numpy as np
import os


def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = '/scratch/tmp/lterfehr/models/VST/RGBD_VST/pretrained_model/' + 'RGBD_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)

    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(args.test_model_dir))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(args.test_model_dir))


    test_dataset = get_loader(args.img_size, rgb_path=args.rgb, depth_path=args.depth, mode='test')

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
    print('''
                Starting testing:
                    RGB Path: {}
                    Depth Path: {}
                    Testing size: {}
                '''.format(args.rgb, args.depth, len(test_loader.dataset)))

    torch.cuda.synchronize()
    start_time = time.time()

    for i, data_batch in enumerate(test_loader):
        print('Testing process: {}/{}'.format(i + 1, len(test_loader.dataset)))
        images, depths, image_w, image_h, image_path = data_batch
        images, depths = Variable(images.cuda()), Variable(depths.cuda())

        outputs_saliency, outputs_contour = net(images, depths)

        mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

        image_w, image_h = int(image_w[0]), int(image_h[0])

        output_s = F.sigmoid(mask_1_1)

        output_s = output_s.data.cpu().squeeze(0)

        transform = trans.Compose([
            transforms.ToPILImage(),
            trans.Scale((image_w, image_h))
        ])
        output_s = transform(output_s)

        filename = image_path[0].split('/')[-1].split('.')[0]

        # save saliency maps
        if not os.path.exists(args.target):
            os.makedirs(args.target)
        output_s.save(os.path.join(args.target, filename + '.png'))

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    time_per_image = total_time / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    images_processed = len(test_loader.dataset)

    eval_path = os.path.join(args.target, 'evaluation.txt')
    with open(eval_path, 'w') as f:
        f.write(f'Total images processed: {images_processed}\n')
        f.write(f'Total testing time: {total_time:.4f} seconds\n')
        f.write(f'Time per image: {time_per_image:.4f} seconds\n')





