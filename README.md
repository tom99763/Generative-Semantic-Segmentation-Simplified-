## Generative-Semantic-Segmentation-Simplified

### Changes
* The two small CNNs (non-linear) are utilized in the posterior learning stage to map the mask to rgb color space and map back to categorical space.

* ResNet is used as the image encoder instead of swin transformer

```
#linear map
        self.mask2rgb = nn.Sequential(
            nn.Conv2d(self.num_classes, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

        #non linear map
        self.rgb2mask = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 1)
        )
```



### Reference
* https://github.com/fudan-zvg/GSS
