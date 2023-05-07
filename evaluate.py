import os
import time
#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
from tqdm import tqdm
#图像处理库PIL的Image
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from toolbox import get_dataset
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt


def evaluate(logdir, save_predict=False):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')

    testset = get_dataset(cfg)[-1]
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'])

    model = get_model(cfg).to(device)
    model = load_ckpt(logdir, model,kind='best')

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    with torch.no_grad():
        model.eval()
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            # resize
            h, w = image.size(2), image.size(3)
            image = F.interpolate(image, size=(int((h // 32) * 32), int(w // 32) * 32), mode='bilinear',
                                  align_corners=True)
            depth = F.interpolate(depth, size=(int((h // 32) * 32), int(w // 32) * 32), mode='bilinear',
                                  align_corners=True)
            predict = model(image, depth)
            # print(predict.size())
            # return to the original size
            predict = F.interpolate(predict, size=(h, w), mode='bilinear', align_corners=True)

            predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
            label = label.cpu().numpy()
            running_metrics_val.update(label, predict)

            time_meter.update(time.time() - time_start, n=image.size(0))

            if save_predict:
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                predict = class_to_RGB(predict, N=len(testset.cmap), cmap=testset.cmap)  # 如果数据集没有给定cmap,使用默认cmap
                predict = Image.fromarray(predict)
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    # for k, v in metrics[1].items():
    #     print(k, v)
    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)
    #return metrics[0]['mIou: ']


if __name__ == '__main__':
    #parse自带的命令行参数解析包，可以用来方便地读取命令行参数，当你的代码需要频繁地修改参数的时候，使用这个工具可以将参数和代码分离开来
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, help="run logdir", default="run/2022-02-21-09-25/")
    parser.add_argument("-s", type=bool, default=False, help="save predict or not")
    args = parser.parse_args()
    # print(args.logdir)
    evaluate(args.logdir, save_predict=args.s)

