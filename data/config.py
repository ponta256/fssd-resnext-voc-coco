# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)


# SSD512 CONFIGS
# [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
vocd512 = {
    'num_classes': 21,
    'lr_steps': (150, 200, 250),    
    'max_epoch': 250,        
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'box': [6, 6, 6, 6, 6, 4, 4],
    'aspect_ratios': [[2,3], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

vocd512f = {
    'num_classes': 21,
    'lr_steps': (100, 130, 150),    
    'max_epoch': 150,  
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'box': [6, 6, 6, 6, 6, 4, 4],
    'aspect_ratios': [[2,3], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# SSD512 CONFIGS
# [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
voc512 = {
    'num_classes': 21,
    'lr_steps': (150, 200, 250),    
    'max_epoch': 250,        
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# SSD300 CONFIGS
# 'lr_steps': (80000, 100000, 120000),
# 'max_iter': 120000,
voc = {
    'num_classes': 21,
    'lr_steps': (150, 200, 250),    
    'max_epoch': 250,    
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

cocod512 = {
    'num_classes': 81,
    'lr_steps': (100, 130, 150),
    'max_epoch': 150,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35, 76, 153, 230, 307, 384, 460],
    'max_sizes': [76, 153, 230, 307, 384, 460, 537],
    'box': [6, 6, 6, 6, 6, 4, 4],
    'aspect_ratios': [[2,3], [2,3], [2,3], [2,3], [2,3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

'''
coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
'''
