"""  
CRAFT detector implementation.
Not written by me.
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

from vgg16_bn import vgg16_bn, init_weights

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

    def detect(self, img, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
        """
        Detect text regions in an image.
        Returns list of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Preprocess image
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(next(self.parameters()).device)
        
        with torch.no_grad():
            y, _ = self.forward(img)
        
        # Post-process to get bounding boxes
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()
        
        # Simple thresholding - you may want more sophisticated post-processing
        boxes = self._get_boxes(score_text, score_link, text_threshold, link_threshold)
        return boxes
    
    def _get_boxes(self, score_text, score_link, text_threshold, link_threshold):
        """Extract bounding boxes from score maps."""
        import cv2
        import numpy as np
        
        text_score = (score_text > text_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(text_score, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Filter tiny boxes
                boxes.append((x, y, x + w, y + h))
        return boxes
    
    def crop_and_resize(self, img, box, target_height=32):
        """Crop and resize text region to fixed height."""
        x1, y1, x2, y2 = box
        
        if isinstance(img, torch.Tensor):
            crop = img[:, y1:y2, x1:x2]
        else:
            crop = transforms.ToTensor()(img)[:, y1:y2, x1:x2]
        
        # Resize to fixed height, variable width
        h, w = crop.shape[1], crop.shape[2]
        new_w = int(w * target_height / h)
        crop = F.interpolate(crop.unsqueeze(0), size=(target_height, new_w), mode='bilinear', align_corners=False)
        return crop.squeeze(0)

if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)
