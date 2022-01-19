# Visdom Tutorial

This is a simple tutorial to start using [Visdom][1] to plot graphs when using PyTorch. 


## Install

Install Visdom using pip

```pip install visdom```

or Anaconda

```conda install -c conda-forge visdom ```

## Start the server

To use vidsom, you need to start the visdom server first. Make sure the server is running before you run your algorithms! 

Start the server with:

```python -m visdom.server```

Then, in your browser, you can go to:

```http://localhost:8097``` 

You will see the visdom interface:

<p align="center">
<img src="https://github.com/noagarcia/visdom-tutorial/blob/master/visdom-main.png" alt="visdom" width="460"/>
</p>


## Let's plot something!

This repository contains the code to train a simple image classifier ([tutorial][2]) and to produce some example plots with visdom.


### VisdomLinePlotter
We first create a visdom object to make the calls for us. You can find this code in ```utils.py``` file:

```python
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
```
 
To create a new plot or to add new data to an existing plot, we will call ```plot(var_name, split_name, title_name, x, y)```, with:
- ```var_name```: variable name (e.g. ```loss```, ```acc```)
- ```split_name```: split name (e.g. ```train```, ```val```)
- ```title_name```: titles of the graph (e.g. ```Classification Accuracy```)
- ```x```: x axis value (e.g. epoch number)
- ```y```: y axis value (e.g. epoch loss)


### Main process

In ```train.py``` we create the visdom object as a global variable:

```python
import utils

if __name__ == "__main__":
    
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='Tutorial Plots')
```

Now we can use ```plotter``` in any function to add data to our graphs.

In the training function we add the loss value after every epoch as:

```python
plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)
```

In the validation function we add the loss and the accuracy values as:

```python
plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)
```

And that's it! Simple, isn't it? 

Now we can check the graphs at any point by visiting the visdom interface at http://localhost:8097 as long as the visdom server is still running.

<p align="center">
<img src="https://github.com/noagarcia/visdom-tutorial/blob/master/visdom-plots.png" alt="visdom" width="460"/>
</p>




[1]: https://github.com/facebookresearch/visdom
[2]: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html