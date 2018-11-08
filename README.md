# ssrnet2caffe
基于keras的SSRNet转换到Caffe框架

### Reference：

#### SSR-Net
https://github.com/shamangary/SSR-Net

#### ssrnet2caffe
https://github.com/nerddd/SSRNet-caffe

转换的时候存在参数对应不上的问题，原因是TensorFlow和Caffe的机制不同，在poolling层的源码实现上，Caffe有对结果向上取整的操作，最终导致TF下和Caffe下特征图的大小不一致，可参看我的一篇博客（https://blog.csdn.net/lwplwf/article/details/82418110）。

![image](https://github.com/lwplw/repository_image/blob/master/%E9%80%89%E5%8C%BA_010.png)
