## Prerequisites in general
- Linux
- Python 3 (I strong recommend to use [Anaconda](https://anaconda.org/) to set-up your Python environment and build a new conda virtual environment for this project.)
- CUDA==10.0
- Pytorch>=1.0 and corresponding version of torchvision
- Other required 3rd-party packages is listed in [requirements.txt.](requirements.txt)

To successfully use Visdom for real-time visualization, remember to change the IP Address setting in line 22 of "src/util/visualizer.py" to your own IP address.
```
self.vis = visdom.Visdom(server="IP address of your server", port = opt.display_port)

```