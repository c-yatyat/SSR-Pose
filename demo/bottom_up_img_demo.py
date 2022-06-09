import os
from argparse import ArgumentParser
import cv2

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from xtcocotools.coco import COCO

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)

import mmcv
from mmpose.core import make_heatmaps, make_tagmaps


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    assert (dataset == 'BottomUpCocoDataset')

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = True

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = ('backbone', )

    # process each image
    for i in range(len(img_keys)):
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # img = mmcv.imread(image_name, 'color', 'rgb')
        # heatmap_grid = make_heatmaps(img, returned_outputs[0]['heatmap'])
        # cv2.imwrite('heatmap.jpg', heatmap_grid)

        # tagmap_grid = make_tagmaps(img, returned_outputs[0]['tagmap'])
        # cv2.imwrite('tagmap.jpg', tagmap_grid)

        
        
        # kpts = []
        # tags = []

#         color = [
# 'orange':               '#FFA500',
# 'orangered':            '#FF4500',
# 'orchid':               '#DA70D6',
# 'palegoldenrod':        '#EEE8AA',
# 'palegreen':            '#98FB98',
# 'paleturquoise':        '#AFEEEE',
# 'palevioletred':        '#DB7093',
# 'papayawhip':           '#FFEFD5',
# 'peachpuff':            '#FFDAB9',
# 'peru':                 '#CD853F',
# 'pink':                 '#FFC0CB',
# 'plum':                 '#DDA0DD',
# 'powderblue':           '#B0E0E6',
# 'purple':               '#800080',
# 'red':                  '#FF0000',
# '#BC8F8F',
# '#4169E1',
# '#8B4513',
# '#FA8072',
# '#FAA460',
# '#2E8B57',
# '#FFF5EE',
# '#A0522D',
# '#C0C0C0',
# '#87CEEB',
# '#6A5ACD',
# '#708090',
# '#FFFAFA',
# '#00FF7F',
# '#4682B4',
# '#D2B48C',
# '#008080',
# '#D8BFD8',
# '#FF6347',
# '#40E0D0',
# '#EE82EE',
# '#F5DEB3',]

        # for i in range(39):
        #     kpts.append(list(range(17)))
        #     tag = []
        #     for j in range(17):
        #         tag.append(returned_outputs[0]['embedding'][i][j][3])
        #     tags.append(tag)
        
        # plt.scatter(tags, kpts)
        # plt.savefig('embedding.png')

        # tag1 = []
        # kpt1 = list(range(17))
        # for i in range(17):
        #     tag1.append(returned_outputs[0]['embedding'][0][i][3])
        # plt.scatter(tag1, kpt1 ,c='#800080')
        # plt.savefig('embedding1.png')

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)


if __name__ == '__main__':
    main()
