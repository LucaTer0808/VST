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

    # Model laden (multi-GPU state_dict bereinigen)
    model_path = os.path.join(args.save_model_dir, 'RGB_VST.pth')
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # `module.` entfernen
        new_state_dict[name] = v
    
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    # --- AB HIER GEÄNDERT: Nur noch ein Dataset & direktes Output-Verzeichnis ---
    
    # Nutzt jetzt direkt die neuen Argumente aus args
    test_dir_img = args.source  
    save_test_path = args.target 

    test_dataset = get_loader(test_dir_img, "", args.img_size, mode='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print('''
               Starting testing:
                   Dataset Path: {}
                   Testing size: {}
               '''.format(test_dir_img, len(test_loader.dataset)))

    # Zielverzeichnis erstellen, falls es noch nicht existiert
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)


    torch.cuda.synchronize()
    start_time = time.time()

    for i, data_batch in enumerate(test_loader):
        images, image_w, image_h, image_path = data_batch
        images = Variable(images.cuda())

        outputs_saliency, outputs_contour = net(images)
        

        mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

        image_w, image_h = int(image_w[0]), int(image_h[0])

        output_s = F.sigmoid(mask_1_1)
        output_s = output_s.data.cpu().squeeze(0)

        transform = trans.Compose([
            transforms.ToPILImage(),
            trans.Scale((image_w, image_h))
        ])
        output_s = transform(output_s)

        # Dateiname extrahieren (z.B. "bild" aus "/pfad/bild.jpg")
        filename = os.path.basename(image_path[0]).split('.')[0]

        # Speichern im direkt angegebenen Verzeichnis
        output_s.save(os.path.join(save_test_path, filename + '.png'))

    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    time_per_image = total_time / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0

    eval_path = os.path.join(args.target, 'evaluation.txt')
    with open(eval_path, 'w') as f:
        f.write('Image processed: {}\n'.format(len(test_loader.dataset)))
        f.write('Total testing time: {:.4f} seconds\n'.format(total_time))
        f.write('Average time per image: {:.4f} seconds\n'.format(time_per_image))
