import os
import sys
import cv2
import numpy as np
import argparse

sys.path.insert(0, '../lib')
from utils import misc_utils, visual_utils

img_root = 'C:/Users/yinuowang3/Downloads/FOV02/20190416/19/'
def eval_all(args):
    # json file
    assert os.path.exists(args.json_file), "Wrong json path!"
    misc_utils.ensure_dir('outputs')
    records = misc_utils.load_json_lines(args.json_file)[:args.number]
    for record in records:
        dtboxes = misc_utils.load_bboxes(
                record, key_name='dtboxes', key_box='box', key_score='score', key_tag='tag')
        gtboxes = misc_utils.load_bboxes(record, 'gtboxes', 'box')
        dtboxes = misc_utils.xywh_to_xyxy(dtboxes)
        gtboxes = misc_utils.xywh_to_xyxy(gtboxes)
        keep = dtboxes[:, -2] > args.visual_thresh
        dtboxes = dtboxes[keep]
        len_dt = len(dtboxes)
        len_gt = len(gtboxes)
        line = "{}: dt:{}, gt:{}.".format(record['ID'], len_dt, len_gt)
        print(line)
        img_path = img_root + record['ID'] + '.jpg'
        img = misc_utils.load_img(img_path)

        width = img.shape[1]
        height = img.shape[0]
        bboxes=[]
        for i in range(len(dtboxes)):
            one_box = dtboxes[i]
            print(one_box)
            one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                        min(one_box[2], width - 1), min(one_box[3], height - 1)])
            print(one_box)
            x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
            bbox=[x1,y1,x2,y2,'person',float(dtboxes[i][4])]
            bboxes.append(bbox)
        print('bboxes: ', bboxes)

        visual_utils.draw_boxes(img, dtboxes, line_thick=1, line_color='blue')
        # visual_utils.draw_boxes(img, gtboxes, line_thick=1, line_color='white')
        fpath = 'outputs-19/{}.jpg'.format(record['ID'])
        cv2.imwrite(fpath, img)


def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', '-f', default=None, required=True, type=str)
    parser.add_argument('--number', '-n', default=3, type=int)
    parser.add_argument('--visual_thresh', '-v', default=0.3, type=int)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_eval()

    
