import os
import bpy
import bmesh

def load_base_mesh(scene, base_mesh_name = 'jelly'):
    old_objs = set(scene.objects)
    fpath = os.path.join(ROOT_DIR, f'{PREFIX}_{scene.frame_start}.obj')
    bpy.ops.import_scene.obj(filepath=fpath, split_mode = 'OFF')
    obj = list(set(scene.objects) - old_objs)[0]
    obj.name = base_mesh_name
    return obj

def load_shape_keys(scene, base_obj):
    import numpy as np
    class ObjLoader(object):
        def __init__(self, fileName):
            self.vertices = []
            self.faces = []
            try:
                f = open(fileName)
                for line in f:
                    if line[:2] == "v ":
                        vertex = list(map(float, line[1:].split()))
                        self.vertices.append(vertex)
                    elif line[0] == "f":
                        line = line[1:].split()
                        face = []
                        for item in line:
                            if item.find('/') > 0:
                                item = item[:item.find('/')]
                            face.append(int(item)-1)
                        self.faces.append(face)
                f.close()
            except IOError:
                print(f'{fileName} not found.')
        
            self.vertices = np.asarray(self.vertices)
            self.faces = np.asarray(self.faces)
        
        def normalize(self):
            vertices = self.vertices
            max_bound, min_bound = vertices.max(0), vertices.min(0)
            vertices = (vertices - (max_bound+min_bound)/2) / (max_bound - min_bound).max()
            self.vertices = vertices

        def export(self, filename):
            with open(filename, 'w') as f:
                f.write("# OBJ file\n")
                for v in self.vertices:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                for p in self.faces:
                    f.write("f ")
                    for i in p:
                        f.write(f'{i+1} ')
                    f.write("\n")

    bpy.ops.object.select_all(action='DESELECT') 
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.object.shape_key_add()
    bpy.ops.object.mode_set(mode="EDIT")
    for f in range(scene.frame_start + 1, scene.frame_end + 1):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.shape_key_add()
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(base_obj.data)
        obj_mesh = ObjLoader(os.path.join(ROOT_DIR, f'{PREFIX}_{f}.obj'))
        for v in bm.verts:
            v.co = obj_mesh.vertices[v.index]
        bmesh.update_edit_mesh(base_obj.data, True)
        bm.free()
    bpy.ops.object.mode_set(mode="OBJECT")

def load_key_frames(scene, base_obj):
    bpy.ops.object.select_all(action='DESELECT') 
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    key_frame_num = len(base_obj.data.shape_keys.key_blocks)
    for i in range(key_frame_num):
        frame = scene.frame_start + i
        shape_key = base_obj.data.shape_keys.key_blocks[i]
        if i > 0:
            shape_key.value = 0
            shape_key.keyframe_insert(data_path='value', frame = frame - 1)  
        shape_key.value = 1
        shape_key.keyframe_insert(data_path='value', frame = frame)  
        if i < key_frame_num - 1:
            shape_key.value = 0
            shape_key.keyframe_insert(data_path='value', frame = frame + 1) 


if __name__ == '__main__':
	ROOT_DIR = 'example_dir'
	PREFIX = 'ball'
	scene = bpy.context.scene
	obj = load_base_mesh(scene)
	load_shape_keys(scene, obj)
	load_key_frames(scene, obj)