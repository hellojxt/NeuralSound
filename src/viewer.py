import plotly.graph_objs as go
from ipywidgets import interactive, IntSlider, HBox, VBox, Box, Layout, Text, Button, Output, FloatSlider
import numpy as np
import torch

row_layout = Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-around'
            )
col_layout = Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                flex = '1 1 auto'
            )

point_size = 1

def cone(pos, direction, features, length):
    k = (length / 30)
    pos, direction = np.array(pos), np.array(direction)
    p1, p2 = pos - direction*k*0.5, pos - direction*5*k
    r = np.random.rand(3)
    r1 = np.cross(r / sum(r*r)**0.5, direction)
    r2 = np.cross(r1, direction)
    r1, r2 = r1*k, r2*k
    b1, b2, b3, b4 = p2 + r1, p2 + r2, p2 - r1, p2 - r2
    vertices = np.array([p1, b1, b2, b3, b4]).T
    elements = np.array([
        [0,1,2],
        [0,2,3],
        [0,3,4],
        [0,4,1],
        [1,4,2],
        [2,4,3]
    ]).T
    colors = np.ones((features.shape[0], 6))*features.max(1)[...,np.newaxis]
    return vertices, elements, colors

def show(points, colors = None, show_axis = False, title = '', notation = None):
    '''
    Shape of parameters:
    points      [node_num, 3] 
    colors      [node_num, feature_num]
    '''
    if colors is None:
        colors = points
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
        colors = colors.cpu().numpy()
    bbox_max, bbox_min = points.max(0), points.min(0)
    length = (bbox_max - bbox_min).max()
    sphere_vertices = np.array([
        [ 1.,  0.,  0.],
        [-1., -0., -0.],
        [ 0.,  1.,  0.],
        [-0., -1., -0.],
        [ 0.,  0.,  1.],
        [-0., -0., -1.]
    ])*0.015*length*point_size

    sphere_elements = np.array([
        [2, 4, 0],
        [1, 4, 2],
        [3, 4, 1],
        [0, 4, 3],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3]]
    )
    vertices, elements, colors = points2sphere(points, colors.T, sphere_vertices, sphere_elements)
    if notation is not None:
        pos, direction = notation
        vertices_cone, elements_cone, colors_cone = cone(pos, direction, colors, length)
        elements_cone += vertices.shape[-1]
        vertices = np.concatenate((vertices, vertices_cone), axis=1)
        elements = np.concatenate((elements, elements_cone), axis=1)
        colors = np.concatenate((colors, colors_cone), axis=1)

    #return viewer(vertices_cone, elements_cone, colors_cone, show_axis, title).box
    return viewer(vertices, elements, colors , show_axis, title).box

def save_img(points, colors, filename):
    '''
    Shape of parameters:
    points      [node_num, 3] 
    colors      [node_num, feature_num]
    '''
    if colors is None:
        colors = points
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
        colors = colors.cpu().numpy()
    bbox_max, bbox_min = points.max(0), points.min(0)
    length = (bbox_max - bbox_min).max()
    sphere_vertices = np.array([
        [ 1.,  0.,  0.],
        [-1., -0., -0.],
        [ 0.,  1.,  0.],
        [-0., -1., -0.],
        [ 0.,  0.,  1.],
        [-0., -0., -1.]
    ])*0.015*length

    sphere_elements = np.array([
        [2, 4, 0],
        [1, 4, 2],
        [3, 4, 1],
        [0, 4, 3],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3]]
    )
    vertices, elements, colors = points2sphere(points, colors.T, sphere_vertices, sphere_elements)
    layout = go.Layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    fig = go.Figure(layout=layout)
    vv = viewer(vertices, elements, colors , False, title='')
    fig.add_trace(vv.fig.data[0])
    scene=dict(
        xaxis=dict(showticklabels=False,visible=False),
        yaxis=dict(showticklabels=False,visible=False),
        zaxis=dict(showticklabels=False,visible=False),
    )
    fig.update_layout(scene=scene)
    size = length*0.04
    camera = dict(
        eye=dict(x=size, y=size, z=size)
    )

    fig.update_layout(scene_camera=camera)
    fig.write_image(filename)
    vv.clear()
    

def points2sphere(points, colors, sphere_vertices, sphere_elements):
    '''
    Shape of parameters:
    points      [node_num, 3] 
    colors      [feature_num, node_num]
    '''
    vertices = points[:,np.newaxis,:] + sphere_vertices
    vertices = vertices.reshape(-1,3)
    offsets = len(sphere_vertices)*np.arange(len(points))
    elements = offsets[...,np.newaxis,np.newaxis] + sphere_elements
    elements = elements.reshape(-1,3)
    colors = colors[...,np.newaxis]*np.ones(len(sphere_elements))
    colors = colors.reshape(colors.shape[0], -1)
    return vertices.T, elements.T, colors

class viewer():
    def __init__(self, vertices, elements, data, show_axis, title):
        '''
        Shape of parameters:
        vertices    [3, node_num]
        elements    [3, triangle_num]
        data        [feature_num, triangle_num]
        '''
        self.title = title
        self.show_axis = show_axis
        self.vertices, self.elements, self.data = vertices, elements, data
        self._in_batch_mode = False
        self.items = [
                self.init_data_selector(),
                self.init_3D()
            ]

    @property
    def box(self):
        return Box(
            self.items,
            layout=col_layout
        )
    def clear(self):
        self.fig.data[0].x = []
        self.fig.data[0].y = []
        self.fig.data[0].z = []
        self.fig.data[0].i = []
        self.fig.data[0].j = []
        self.fig.data[0].k = []
        self.fig.data[0].intensity = []


    def init_3D(self):
        bound_max = self.vertices.max()
        bound_min = self.vertices.min()
        self.fig = go.FigureWidget(data = [
                    go.Mesh3d(
                            x=self.vertices[0],
                            y=self.vertices[1],
                            z=self.vertices[2],
                            intensity = self.data[0],
                            intensitymode='cell',
                            colorscale='Jet',
                            i = self.elements[0],
                            j = self.elements[1],
                            k = self.elements[2],
                            # showlegend=self.show_axis, 
                            # showscale=self.show_axis,
                        )
                ]
            )
        self.fig.layout.height = 300
        self.fig.layout.width = 400
        self.fig.layout.autosize = False
        self.fig.layout.title = self.title
        self.fig.update_layout(
                margin=dict(l=0,r=0,b=0,t=0,pad=0),
                title={'y':0.9,'x':0.4},
        )
        self.fig.update_layout(scene_aspectmode='cube')
        if self.show_axis:
            scene=dict(
                xaxis=dict(range=[bound_min,bound_max]),
                yaxis=dict(range=[bound_min,bound_max]),
                zaxis=dict(range=[bound_min,bound_max]),
            )
        else:
            scene=dict(
                xaxis=dict(showticklabels=False,visible=False,range=[bound_min,bound_max]),
                yaxis=dict(showticklabels=False,visible=False,range=[bound_min,bound_max]),
                zaxis=dict(showticklabels=False,visible=False,range=[bound_min,bound_max]),
            )
        self.fig.update_layout(scene=scene)
        self.fig.data[0].lighting = {
            "ambient": 0.7,
            "diffuse": 1,
            "specular": 0.3,
            "roughness": .5,
        }
        self.fig.update_layout(title_font_size=10)
        return Box([self.fig], layout = row_layout)


    def init_data_selector(self):
        self.int_range = IntSlider(min=0, max=len(self.data)-1, step=1, value=0,layout = Layout(flex = '3 1 auto'), description="Index")
        def select(change):
            self.fig.data[0].intensity = self.data[self.int_range.value]
        self.int_range.observe(select, names='value')
        return Box([self.int_range], layout = row_layout)

def show_boxs(*args):
    lst = [box for box in args]
    return Box(lst, layout = row_layout)