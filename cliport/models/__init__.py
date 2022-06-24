from cliport.models.resnet import ResNet43_8s
from cliport.models.resnet_lat import ResNet45_10s
from cliport.models.clip_lingunet_lat import CLIPLingUNetLat
from cliport.models.resnet_lang import ResNet43_8s_lang
from cliport.models.clip_lingunet import CLIPLingUNet
names = {
    'plain_resnet': ResNet43_8s,
    'plain_resnet_lat': ResNet45_10s,
    'plain_resnet_lang': ResNet43_8s_lang,
    'clip_lingunet_lat': CLIPLingUNetLat,
    'clip_lingunet': CLIPLingUNet
}
