import dynet_config
dynet_config.set(autobatch=True)
import dynet as dy
import numpy as np
from time import time

class Module:
    def __new__(cls, *args, **kwargs):
        if 'parent' in kwargs or 'is_top' in kwargs and kwargs['is_top']:
            return super(Module, cls).__new__(cls)
        else:
            return cls.create(*args, **kwargs)
        
    def __init__(self, parent=None, name=None, **kwargs):
        if parent: self.params = parent.add_child(self, name)
        else:      self.params = dy.ParameterCollection()
        self.children = []
        self._training = True
        
    @classmethod    
    def create(cls, *args, **kwargs):
        assert 'parent' not in kwargs
        def init(parent, name=None):
            assert isinstance(parent, Module)
            return cls(*args, **kwargs, parent=parent, name=name)
        return init

    def __setattr__(self, k, v):
        try:
            if v.__qualname__ == 'Module.create.<locals>.init':
                _module = v(parent=self, name=k)
                super(Module, self).__setattr__(k, _module)
                return
        except AttributeError:
            pass
        super(Module, self).__setattr__(k, v)
    def save(self, fname):
        self.params.save(fname)
    def load(self, fname):
        self.params.populate(fname)
    def add_child(self, child, name):
        self.children.append(child)
        if name:
            assert name not in self.__dict__, f"sub module name {name} already exists!!"
            self.__dict__[name] = child
        return self.params.add_subcollection(name)
    def __repr__(self):
        return '\n'.join(repr(child) for child in self.children)
    def train(self): 
        self.training = True
        for child in self.children: child.train()
    def eval(self):  
        self.training = False
        for child in self.children: child.eval()

class Linear(Module):
    def __init__(self, n_in, n_out, activ=dy.rectify, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.n_in, self.n_out, self.activ = n_in, n_out, activ
        self.w = self.params.add_parameters((n_out, n_in), init='he', name='weights')
        self.b = self.params.add_parameters(n_out, init=0, name='bias')
        self.activ = self.activ
        
    def __call__(self, x):
        try:
            if self.activ: return self.activ(self.w*x + self.b)
            else:          return self.w*x + self.b
        except NotImplementedError:
            return [self(_x) for _x in x]
    def __repr__(self):
        return f"Linear layer: in={self.n_in}, out={self.n_out}, activation={self.activ.__name__ if self.activ else 'no_op'}"
    
class Sequential(Module):
    def __init__(self, *layer_gens, parent=None, name=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.layers = [l(parent=self) if l.__qualname__=='Module.create.<locals>.init' else l for l in layer_gens ]
        
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x

class Input(Module):
    def __init__(self, parent): super.__init__(parent)
    def __call__(self, x):      return dy.inputVector(x)
    
class SimpleModel(Module):
    def __new__(cls, *args, **kwargs):
        return super(cls, SimpleModel).__new__(cls, is_top=True, **kwargs)
    def __init__(self, n_in, n_hid, n_out, parent=None, **kwargs):
        super().__init__(parent)
        self.lin1 = Linear(n_in, n_hid)
        Linear(n_hid, n_out, activ=None, parent=self, name='lin2')
        
    def __call__(self, x):
        x = dy.inputTensor(x)
        return self.lin2(self.lin1(x))