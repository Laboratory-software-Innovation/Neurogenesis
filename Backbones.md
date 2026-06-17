# 📚 NeuroGenesis Backbone Networks

The curated list of architectures employed in NeuroGenesis.


## FNET
Mixing tokens using Fourier Transforms instead of self-attention.
```
@article{lee2021fnet,
  title={Fnet: Mixing tokens with fourier transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```

## gMLP
Attention-free MLP-based architecture using spatial gating.
```
@article{liu2021pay,
  title={Pay attention to mlps},
  author={Liu, Hanxiao and Dai, Zihang and So, David and Le, Quoc V},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={9204--9215},
  year={2021}
}
```

## MLP Mixer
An all-MLP architecture designed for vision tasks.
```
@article{tolstikhin2021mlp,
  title={Mlp-mixer: An all-mlp architecture for vision},
  author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={24261--24272},
  year={2021}
}
```

## Transformer
### (1) Original Transformer
Introduced the concept of self-attention for sequence modeling.
```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
### (2) Vision Transformer (ViT)
Adapted the Transformer architecture for image classification.
```
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## External Transformer
```
@article{guo2022beyond,
  title={Beyond self-attention: External attention using two linear layers for visual tasks},
  author={Guo, Meng-Hao and Liu, Zheng-Ning and Mu, Tai-Jiang and Hu, Shi-Min},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={45},
  number={5},
  pages={5436--5447},
  year={2022},
  publisher={IEEE}
}
```

## ConvMixer
Combines convolutional operations with token mixing.
```
@article{trockman2022patches,
  title={Patches are all you need?},
  author={Trockman, Asher and Kolter, J Zico},
  journal={arXiv preprint arXiv:2201.09792},
  year={2022}
}
```

## AFTFull

```
@article{zhai2021attention,
  title={An attention free transformer},
  author={Zhai, Shuangfei and Talbott, Walter and Srivastava, Nitish and Huang, Chen and Goh, Hanlin and Zhang, Ruixiang and Susskind, Josh},
  journal={arXiv preprint arXiv:2105.14103},
  year={2021}
}
```

## ResidualAttention
```
@inproceedings{wang2017residual,
  title={Residual attention network for image classification},
  author={Wang, Fei and Jiang, Mengqing and Qian, Chen and Yang, Shuo and Li, Cheng and Zhang, Honggang and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3156--3164},
  year={2017}
}
```

## SEAttention
```
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}
```

## SimAM

## DoubleAttention
```
@article{chen20182,
  title={A\^{} 2-nets: Double attention networks},
  author={Chen, Yunpeng and Kalantidis, Yannis and Li, Jianshu and Yan, Shuicheng and Feng, Jiashi},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}
```

## PerformerAttention
```
@article{choromanski2020rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarlos, Tamas and Hawkins, Peter and Davis, Jared and Mohiuddin, Afroz and Kaiser, Lukasz and others},
  journal={arXiv preprint arXiv:2009.14794},
  year={2020}
}
```


## ParNetAttention, UFOAttention
```
@article{goyal2022non,
  title={Non-deep networks},
  author={Goyal, Ankit and Bochkovskiy, Alexey and Deng, Jia and Koltun, Vladlen},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={6789--6801},
  year={2022}
}
```

## ECAAttention
```
@inproceedings{wang2020eca,
  title={ECA-Net: Efficient channel attention for deep convolutional neural networks},
  author={Wang, Qilong and Wu, Banggu and Zhu, Pengfei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11534--11542},
  year={2020}
}
```

## SwitchTransformer
```
@article{fedus2022switch,
  title={Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={120},
  pages={1--39},
  year={2022}
}
```

## CBAM
```
@inproceedings{woo2018cbam,
  title={Cbam: Convolutional block attention module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={3--19},
  year={2018}
}
```
