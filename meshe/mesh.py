import numpy as np 
import meshio

class Mesh : 

    def __init__(self,dimension = None,type = None):
        '''
        arguments : 
        dimension ::: int ::: dimension of the domain (1, 2 or 3)
        type ::: str ::: elements type 
        '''
        self.dim = dimension
        self.type = type 
        self.elements = None
        self.boundary_elements = None
        self.boundary_tags = None
        self.nodes = None 
        self.internal_faces = None 
        self.internal_connectivity = None 
    
    def _calc_centroid(self,element):
        '''
        arguments 
        element ::: np.array (n_nodes,n_dim) ::: coordinates of the 
        nodes defining the surface element
        returns 
        centroid ::: np.array (n_dim,) ::: centroid of the element
        '''
        centroid = np.mean(element,axis = 0)
        return centroid 

    def _calc_surface_area(self,surface_element):
        '''
        arguments 
        surface_element ::: np.array (n_nodes,3) ::: coordinates of the 
        nodes defining the surface element
        returns 
        surface ::: float ::: surface of that element 
        '''
        centroid = self._calc_centroid(surface_element)
        el_size = np.size(surface_element,0)
        surface = 0
        for i in range(el_size):
            # Get node coordinate in the frame centered 
            # at the element centroid
            p1 = surface_element[i]
            if i == el_size - 1 : 
                p2 = surface_element[0]
            else :
                p2 = surface_element[i+1]
            p1 = p1 - centroid
            p2 = p2 - centroid 
            # Calculate triangle surface (centroid, p1, p2)
            surface_tmp = 0.5*np.sqrt(np.sum(np.cross(p1,p2)**2.))
            # Sum triangles surfaces into the element surface variable
            surface += surface_tmp
        return surface
    
    def _calc_surface_normal(self,surface_element):
        '''
        arguments 
        surface_element ::: np.array (n_nodes,3) ::: coordinates of the 
        nodes defining the surface element
        returns 
        normal_vec ::: np.array(3,) ::: Normal vector of the Surface element
        '''
        centroid = self._calc_centroid(surface_element)
        p1 = surface_element[1]
        p2 = surface_element[0]
        p1 = p1 - centroid
        p2 = p2 - centroid 
        normal_vec = np.cross(p1,p2)
        normal_norm = np.sqrt(np.sum(normal_vec**2.))
        return normal_vec/normal_norm
    
    def _calc_surface_volflux(self,surface_element,cell_centroid): 
        '''
        arguments 
        surface_element ::: np.array (n_nodes,3) ::: coordinates of the 
        nodes defining the surface element
        cell_centroid ::: np.array (3,) ::: coordinates of the control 
        volume centroid. Note that surface_element should be a surface of 
        that control volume. 
        returns 
        volflux ::: float ::: (1/3)*dot(x,n)*Sf
        '''
        face_centroid = self._calc_centroid(surface_element)
        face_normal = self._calc_surface_normal(surface_element)
        face_surface = self._calc_surface_area(surface_element)
        # express face_centroid in the frame centered at cell_centroid 
        face_centroid = face_centroid - cell_centroid
        # orient normal outward of the control volume 
        sign = np.sign(np.dot(face_centroid,face_normal))
        if sign == 1 :
            face_normal = face_normal
        elif sign == - 1 : 
            face_normal = - face_normal
        else : 
            raise ValueError("Error Face normal is perpendicular to face centroid /"
                             "cell centroid vector")
        #
        volflux = (1/3)*np.dot(face_centroid,face_normal)*face_surface
        return volflux
    
    def _calc_element_volume(self,element_surfaces,centroid):
        '''
        arguments 
        element_surfaces ::: np.array (n_surfaces,n_nodes,n_dim) ::: coordinates
        of the 
        nodes defining the surfaces surrounding the element
        centroid ::: np.array (3,) ::: coordinates of the control 
        volume centroid. Note that surface_element should be a surface of 
        that control volume. 
        returns 
        volume ::: float ::: elemnt volume 
        '''
        #centroid = self._calc_centroid(element)
        volume = 0 
        for element_surface in element_surfaces : 
            volflux = self._calc_surface_volflux(element_surface,centroid)
            volume = volume + volflux
        return volume

class TetraMesh(Mesh): 

    def __init__(self):
        '''
        arguments : 
        dimension ::: int ::: dimension of the domain (1, 2 or 3)
        type ::: str ::: elements type 
        '''
        super().__init__(dimension = 3, type = 'tet')

    def gmsh_reader(self,path) : 
        '''
        path ::: str ::: path of the gmsh mesh ('.msh' file)
        '''
        mesh = meshio.read("test.msh")
        self.nodes = mesh.points
        surf_elements = []
        vol_elements = []
        surf_tags = []
        for i in range(len(mesh.cells)):
            cell = mesh.cells[i]
            elements_data = mesh.cell_data['gmsh:physical'][i]
            if cell.type == 'triangle' : 
                surf_elements.append(cell.data)
                surf_tags.append(elements_data)
            if cell.type == 'tetra':
                vol_elements.append(cell.data)
        vol_elements = np.concatenate(vol_elements)
        surf_elements = np.concatenate(surf_elements)
        surf_tags = np.concatenate(surf_tags)
        self.elements = vol_elements
        self.boundary_elements = surf_elements
        self.boundary_tags = surf_tags
    
    def _get_elements_faces(self):
        '''
        returns 
        surfaces ::: np.array (N_surfaces, 3) ::: surfaces elements 
        N_surfaces shall equal 0.5*(4*Nelements+N_bnd_faces)
        '''
        surfaces = []
        # Elements loop 
        for i,element in enumerate(self.elements) : 
            node1, node2, node3, node4 = element 
            face1 = [node1, node2, node3]
            face2 = [node1, node3, node4]
            face3 = [node1, node4, node2]
            face4 = [node2, node3, node4]
            # Check surfaces 
            bool_face1 = True
            bool_face2 = True
            bool_face3 = True
            bool_face4 = True
            if i != 0 :
                surfaces_tmp = np.sort(surfaces, axis = 1).tolist()
                face_tmp = np.sort(face1).tolist()
                if face_tmp in surfaces_tmp : 
                    bool_face1 = False
                face_tmp = np.sort(face2).tolist()
                if face_tmp in surfaces_tmp : 
                    bool_face2 = False
                face_tmp = np.sort(face3).tolist()
                if face_tmp in surfaces_tmp : 
                    bool_face3 = False
                face_tmp = np.sort(face4).tolist()
                if face_tmp in surfaces_tmp : 
                    bool_face4 = False
            # Add surfaces 
            if bool_face1 : 
                surfaces.append(face1)
            if bool_face2 : 
                surfaces.append(face2)
            if bool_face3 : 
                surfaces.append(face3)
            if bool_face4 : 
                surfaces.append(face4)
        surfaces = np.asarray(surfaces)
        print('Number of surfaces :',np.shape(surfaces))
        print(type(surfaces))
        return surfaces
