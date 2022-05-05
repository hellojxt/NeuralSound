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
    
    def get_bounding_box(self):
        vertices = self.vertices
        max_bound, min_bound = vertices.max(0), vertices.min(0)
        return max_bound, min_bound

    def get_length(self):
        max_bound, min_bound = self.get_bounding_box()
        return (max_bound - min_bound).max()
        
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