import numpy as np
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from make_kirplate import make_kirPlate


graphviz = GraphvizOutput(output_file='filter_none.png')

with PyCallGraph(output=graphviz):
    make_kirplate(4,1,0, False, np.array([0]))
