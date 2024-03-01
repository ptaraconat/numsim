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
        # Mesh Elements : filled with node index
        self.elements = None
        self.boundary_elements = None # change name to bndfaces
        self.intfaces = None # change name to intfaces
        # face connectivity : Filled with element index 
        self.intfaces_elem_conn = None # change name to intfaces_elem_conn
        self.boundary_connectivity = None # change name to bndfaces_elem_conn
        # element connectivity : Filled with face index 
        self.elements_intf_conn = None # change name to elements_intf_conn
        self.boundary_tags = None # change name to bndfaces_elem_tags
        # nodes coordinates (x,y,z)
        self.nodes = None 
        # data 
        self.elements_centroids = None 
        self.elements_data = {}
    
    def set_elements_centroids(self):
        '''
        set the elements centroid array 
        '''
        centroids_arr = np.zeros((np.size(self.elements,0),3))
        for i,element in enumerate(self.elements) : 
            element_nodes_coordinates = self.nodes[element]
            centroid = self._calc_centroid(element_nodes_coordinates)
            centroids_arr[i,:] = centroid
        self.elements_centroids = centroids_arr
    
    def set_elements_data(self,dataname, functional):
        '''
        argument 
        dataname ::: string ::: name of the element data 
        functional ::: callable object ::: function of centroids coordinates 
        that defines the data 
        '''
        if self.elements_centroids == None :
            self.set_elements_centroids()
        x_arr = self.elements_centroids[:,0]
        y_arr = self.elements_centroids[:,1]
        z_arr = self.elements_centroids[:,2]
        data_arr = functional(x_arr,y_arr,z_arr)
        self.elements_data[dataname] = data_arr

    def set_internal_faces(self): 
        '''
        Set the internal faces tables and internal faces to element connectivity table 
        '''
        # calculate elements faces and faces connectivity 
        faces, connectivity= self._get_elements_faces()
        # Get index of faces that are connecteed to two elements 
        bool_arr = np.asarray([len(con) for con in connectivity])
        bool_arr = bool_arr == 2
        int_index = np.where(bool_arr)[0]
        ext_index = np.where(np.logical_not(bool_arr))[0]
        # Get internal faces and associated connectivity
        internal_faces = faces[int_index,:]
        internal_faces_connectivity = np.asarray([connectivity[i] for i in int_index])
        self.intfaces = internal_faces
        self.intfaces_elem_conn = internal_faces_connectivity
        print('Number of internal surfaces : ', np.shape(self.intfaces))
    
    def set_elements_intfaces_connectivity(self) : 
        '''
        Set the elements to internal faces connectivity table 
        '''
        elem_intf_conn = [ [] for i in range(np.size(self.elements,0))]
        print(len(elem_intf_conn))
        for surf_ind, intf_to_el in enumerate(self.intfaces_elem_conn) : 
            elem_ind1, elem_ind2 = intf_to_el
            elem_intf_conn[elem_ind1].append(surf_ind)
            elem_intf_conn[elem_ind2].append(surf_ind)
        self.elements_intf_conn = elem_intf_conn
    
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
    
    def _surface_checker_(self, face, surfaces_list,order_list = True):
        '''
        Check if face belongs to surfaces_list 
        arguments 
        face ::: np.array (N_nodes,) ::: Index of nodes defining the face 
        surfaces_list ::: np.array (N_faces,N_nodes) ::: Array containing several faces 
        Each row is associated to one face. The columns stores the index of nodes defining 
        the faces 
        order_list ::: bool ::: whether or not the surfaces_list is already ordered
        '''
        if order_list : 
            surfaces_tmp = np.sort(surfaces_list, axis = 1)
        else : 
            surfaces_tmp = surfaces_list
        face_tmp = np.sort(face)
        bool_tmp = np.all(surfaces_tmp == face_tmp, axis = 1)
        if np.any(bool_tmp): 
            bool_face = False
            face_paired_index = np.where(bool_tmp)[0][0]
        else :
            bool_face = True
            face_paired_index = None 
        return bool_face, face_paired_index

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
        print(mesh)
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
        surfaces_connectivity ::: np.array (N_surfaces, 2) ::: Elements index to 
        which the surface belong to 
        elements_face_connectivity ::: np.array(N_elements, 4) ::: index of faces 
        that bound the mesh elements 
        '''
        surfaces = []
        surfaces_connectivity = []
        elements_face_connectivity = []
        faces_count = 0 
        # Elements loop 
        for elem_index,element in enumerate(self.elements) : 
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
            if elem_index != 0 :
                surfaces_tmp = np.sort(surfaces, axis = 1)
                bool_face1, face1_paired_index = self._surface_checker_(face1,
                                                                        surfaces_tmp, 
                                                                        order_list= False)
                bool_face2, face2_paired_index = self._surface_checker_(face2,
                                                                        surfaces_tmp, 
                                                                        order_list= False)
                bool_face3, face3_paired_index = self._surface_checker_(face3,
                                                                        surfaces_tmp, 
                                                                        order_list= False)
                bool_face4, face4_paired_index = self._surface_checker_(face4,
                                                                        surfaces_tmp, 
                                                                        order_list= False)
            # Add surfaces 
            if bool_face1 : 
                surfaces.append(face1)
                faces_count +=1
                surfaces_connectivity.append([elem_index])
            else : 
                surfaces_connectivity[face1_paired_index].append(elem_index)
            if bool_face2 : 
                surfaces.append(face2)
                faces_count +=1
                surfaces_connectivity.append([elem_index])
            else : 
                surfaces_connectivity[face2_paired_index].append(elem_index)
            if bool_face3 : 
                surfaces.append(face3)
                faces_count +=1
                surfaces_connectivity.append([elem_index])
            else : 
                surfaces_connectivity[face3_paired_index].append(elem_index)
            if bool_face4 : 
                surfaces.append(face4)
                faces_count +=1
                surfaces_connectivity.append([elem_index])
            else : 
                surfaces_connectivity[face4_paired_index].append(elem_index)
        surfaces = np.asarray(surfaces)
        print('Number of surfaces :',np.shape(surfaces))
        print('Number of surfaces :',faces_count)
        return surfaces, surfaces_connectivity
