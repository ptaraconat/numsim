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
        self.bndfaces = None 
        self.intfaces = None 
        # face connectivity : Filled with element index 
        self.intfaces_elem_conn = None 
        self.bndfaces_elem_conn = None 
        # element connectivity : Filled with face index 
        self.elements_intf_conn = None 
        self.elements_bndf_conn = None
        self.bndfaces_tags = None 
        # nodes coordinates (x,y,z)
        self.nodes = None 
        # data 
        self.elements_centroids = None 
        self.bndfaces_centroids = None 
        self.elements_data = {}
        self.bndfaces_data = {}
        self.physical_entities = None
        
    def save_vtk(self,output_file = 'dump.vtk'):
        '''
        save the mesh into vtk format 
        arguments 
        output_file ::: str ::: name of the dumped file 
        '''
        # Create meshio.Mesh object
        data = {}
        for key,val in self.elements_data.items():
            data[key] = [val.tolist()]
        mesh = meshio.Mesh(
            points=self.nodes,
            cells=[("tetra", self.elements)],
            cell_data=data
        )
        meshio.write(output_file, mesh, file_format="vtk")
        
    def _get_bc_index(self,bc_name): 
        '''
        arguments 
        bc_name ::: str ::: name of the boundary condition
        returns 
        bc_index ::: int ::: index label of that boundary, if if exists
        '''
        return self.physical_entities[bc_name][0]
        
    def _get_boundary_elements_index(self, N = 6):
        '''
        returns 
        index_list ::: list of int ::: indexes of elements located at a boundary
        '''
        index_list = [i for i,el_conn in enumerate(self.elements_intf_conn) if  len(el_conn) < N]
        return index_list
        
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
        if np.all(self.elements_centroids == None) :
            self.set_elements_centroids()
        x_arr = self.elements_centroids[:,0]
        y_arr = self.elements_centroids[:,1]
        z_arr = self.elements_centroids[:,2]
        data_arr = functional(x_arr,y_arr,z_arr)
        self.elements_data[dataname] = data_arr
    
    def set_bndfaces_centroids(self):
        '''
        set the doundary faces centroid array 
        '''
        centroids_arr = np.zeros((np.size(self.bndfaces,0),3))
        for i,face in enumerate(self.bndfaces) : 
            face_nodes_coordinates = self.nodes[face]
            centroid = self._calc_centroid(face_nodes_coordinates)
            centroids_arr[i,:] = centroid
        self.bndfaces_centroids = centroids_arr
    
    def set_bndfaces_data(self,dataname, functional):
        '''
        argument 
        dataname ::: string ::: name of the element data 
        functional ::: callable object ::: function of centroids coordinates 
        that defines the data 
        '''
        if np.all(self.bndfaces_centroids == None) :
            self.set_bndfaces_centroids()
        x_arr = self.bndfaces_centroids[:,0]
        y_arr = self.bndfaces_centroids[:,1]
        z_arr = self.bndfaces_centroids[:,2]
        data_arr = functional(x_arr,y_arr,z_arr)
        self.bndfaces_data[dataname] = data_arr
        
    def set_bndfaces_data_from_bc(self,dataname,dict):
        '''
        arguments 
        dataname ::: string ::: name of the element data
        dict ::: dictionnary ::: dictionary of uniform boundary condition
        '''
        # Init bndface_data with None 
        self.bndfaces_data[dataname] = np.asarray([None for _ in range(np.size(self.bndfaces,0))])
        # Loop over different boundary conditions 
        for bc_key,val in dict.items():
            bc_index = self._get_bc_index(bc_key)
            type = val['type']
            bc_val = val['value']
            # get index associated with the current bondary condition 
            surfaces_indices =np.squeeze(np.argwhere(self.bndfaces_tags == bc_index))
            if surfaces_indices.shape == () : 
                surfaces_indices = [surfaces_indices]
            # Loop over the later boundary surfaces 
            if type == 'dirichlet':
                for i in surfaces_indices :
                    self.bndfaces_data[dataname][i] = bc_val
 
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

    def set_boundary_faces(self): 
        '''
        Brute force strategy. May be optimized 
        Set the boundary faces to element connectivity table, given the bndfaces table/attribute
        '''
        count = 0
        el_bndf_conn = [[] for _ in range(np.size(self.elements,0)) ]
        bndf_el_conn = np.zeros((np.size(self.bndfaces,0),1))
        elbnd_index = self._get_boundary_elements_index()
        for bndface_ind,bndface in enumerate(self.bndfaces) : 
            for i in elbnd_index:
                element = self.elements[i]
                elem_faces = np.asarray(self._get_element_faces(element))
                bool_face, face_paired_index = self._surface_checker_(bndface, 
                                                                      elem_faces,
                                                                      order_list = True)
                if not(bool_face) :
                    el_bndf_conn[i].append(bndface_ind)
                    bndf_el_conn[bndface_ind] = i
                    count += 1
        self.elements_bndf_conn = el_bndf_conn
        self.bndfaces_elem_conn = bndf_el_conn
        print(count)

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
        
    def _calc_face_pairnode_intersection(self,face,node1,node2):
        '''
        arguments 
        face ::: np.array(n_node,3) ::: coordinates of nodes defining 
        the faces (n_node >= 3). 
        node1 ::: np.array(3,) ::: coordinates of node1
        node2 ::: np.array(3,) ::: coordinates of node2
        returns 
        intersection_vertex ::: np.array(3,) :::
        '''
        face_normal = self._calc_surface_normal(face)
        point_on_plane = np.squeeze(face[0,:])
        vector_start = node1
        vector_end = node2
        # Calculate the direction vector of the line
        vector_direction = vector_end - vector_start
        # Calculate the parameter t for the vector equation (intersection point)
        t = np.dot(face_normal, (point_on_plane - vector_start)) / np.dot(face_normal, vector_direction)
        # Calculate the intersection point
        intersection_vertex = vector_start + t * vector_direction
        return intersection_vertex
        
        
    def _calc_face_pairnode_theta(self,face,node1,node2):
        '''
        arguments 
        face ::: np.array(n_node,3) ::: coordinates of nodes defining 
        the faces (n_node >= 3). 
        node1 ::: np.array(3,) ::: coordinates of node1
        node2 ::: np.array(3,) ::: coordinates of node2
        returns 
        angle ::: float :: angle (in radians) between the face normal and
        the node1/node2 direction. 
        '''
        pairnode_vec = node1 - node2 
        face_normal = self._calc_surface_normal(face)
        dot_prod = np.dot(pairnode_vec,face_normal)
        # Calculate the magnitudes of the vectors
        pairnode_vec_mag = np.linalg.norm(pairnode_vec)
        face_normal_mag = np.linalg.norm(face_normal)
        # Calculate the cosine of the angle
        cosine_angle = dot_prod / (pairnode_vec_mag * face_normal_mag)
        # Calculate the angle in radians
        angle = np.arccos(np.abs(cosine_angle))
        return angle
        
    def _calc_vertex_face_distance(self,vertex,face):
        '''
        arguments 
        vertex ::: np.array (3,) ::: coordinates of the vertex
        face ::: np.array(n_node,3) ::: coordinates of nodes defining 
        the faces (n_node >= 3). 
        returns 
        distance ::: float ::: distance between vertex and plane (shortest distance)
        '''
        unit_normal = self._calc_surface_normal(face)
        p1 = face[0]
        vec_tmp = vertex - p1
        distance = np.dot(vec_tmp, unit_normal)
        return np.abs(distance)
        
    
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
        order_list ::: bool ::: whether or not the surfaces_list has to be ordered
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

class Mesh1D(Mesh):
    
    def __init__(self,dx,n_elem,):
        '''
        '''
        super().__init__(dimension = 1, type = 'hexa')
        # Init first element 
        nodes_list = [np.array([0, 0 ,0]),
                    np.array([0, dx, 0]),
                    np.array([0, dx, dx]),
                    np.array([0, 0, dx]),
                    np.array([dx, 0 ,0]),
                    np.array([dx, dx, 0]),
                    np.array([dx, dx, dx]),
                    np.array([dx, 0, dx])]
        elements_list = [np.array([0,1,2,3,4,5,6,7])]
        intfaces_list = []
        bndfaces_list = [np.array([0,1,2,3]),
                        np.array([0,4,7,3]),
                        np.array([1,5,6,2]),
                        np.array([3,7,6,2]),
                        np.array([0,4,5,1])]
        bndfaces_tags_list = [1,2,2,2,2]    
        elements_bndf_conn_list = [[0,1,2,3,4]]
        intfaces_elem_conn_list = []
        bndfaces_elem_conn_list = [0,0,0,0,0]
        for i in range(n_elem-1): 
            #print(i)
            #print('add element', i+2)
            # add dx (along x axis) to previous 4 nodes 
            delta = np.array([dx,0,0])
            last_nodes = nodes_list[-4:]
            add_nodes = [node + delta for node in last_nodes]
            nodes_list += add_nodes
            # add new element 
            last_el = elements_list[-1]
            add_el = [last_el+4]
            elements_list +=add_el
            # add new internal face 
            add_intf = last_el[-4:]
            intfaces_list += [add_intf]
            add_intf_el_conn = [[len(elements_list)-2,len(elements_list)-1]]
            intfaces_elem_conn_list += add_intf_el_conn
            # add new bndfaces 
            last_bndfaces = bndfaces_list[-4:]
            add_bndfaces = [bndface + 4 for bndface in last_bndfaces]
            bndfaces_list +=add_bndfaces
            bndfaces_tags_list += [2,2,2,2]
            elem_id = len(elements_list)-1
            add_bndf_el_conn = [elem_id,elem_id,elem_id,elem_id]
            bndfaces_elem_conn_list += add_bndf_el_conn
            # elem_bndf_conn 
            last_elem_bndf_conn = elements_bndf_conn_list[-1][-4:]
            add_elem_bndf_conn = np.asarray(last_elem_bndf_conn)+4
            elements_bndf_conn_list += [add_elem_bndf_conn.tolist()]
        # Add outlet to bndfaces_list
        outlet_bndface = [elements_list[-1][-4:]]
        bndfaces_list += outlet_bndface
        bndfaces_tags_list += [3]
        bndfaces_elem_conn_list += [elem_id]
        elements_bndf_conn_list[-1].append(len(bndfaces_list)-1)
        #
        self.nodes = np.asarray(nodes_list)
        self.elements = np.asarray(elements_list)
        self.bndfaces = np.asarray(bndfaces_list)
        self.intfaces = np.asarray(intfaces_list)
        self.intfaces_elem_conn = np.asarray(intfaces_elem_conn_list)
        self.elements_bndf_conn = elements_bndf_conn_list
        self.bndfaces_elem_conn = np.expand_dims(np.asarray(bndfaces_elem_conn_list),axis = 1)
        self.bndfaces_tags = np.asarray(bndfaces_tags_list)
        #
        self.set_elements_intfaces_connectivity()
        self.set_elements_centroids()
        

class HexaMesh(Mesh) : 
    
    def __init__(self):
        '''
        arguments : 
        dimension ::: int ::: dimension of the domain (1, 2 or 3)
        type ::: str ::: elements type 
        '''
        super().__init__(dimension = 3, type = 'hexa')
    
    def _get_element_faces(self,element):
        '''
        argument 
        element ::: np.array(n_nodes,) ::: array containing the nodes index that 
        defines the element 
        returns 
        faces ::: list of lists of int ::: list of faces. Each face being defined 
        as a list of nodes index
        '''
        node1, node2, node3, node4 = element 
        face1 = [node1, node2, node3, node4]
        face2 = []
        face3 = []
        face4 = []
        face5 = []
        face6 = []
        faces = [face1, face2, face3, face4, face5, face6]
        return faces
    
    def _get_boundary_elements_index(self):
        '''
        returns 
        index_list ::: list of int ::: indexes of elements located at a boundary
        '''
        index_list = [i for i,el_conn in enumerate(self.elements_intf_conn) if  len(el_conn) < 6]
        return index_list
        
    
    

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
        self.bndfaces = surf_elements
        self.bndfaces_tags = surf_tags
        self.physical_entities = mesh.field_data

    def _get_element_faces(self,element):
        '''
        argument 
        element ::: np.array(n_nodes,) ::: array containing the nodes index that 
        defines the element 
        returns 
        faces ::: list of lists of int ::: list of faces. Each face being defined 
        as a list of nodes index
        '''
        node1, node2, node3, node4 = element 
        face1 = [node1, node2, node3]
        face2 = [node1, node3, node4]
        face3 = [node1, node4, node2]
        face4 = [node2, node3, node4]
        faces = [face1, face2, face3, face4]
        return faces
    
    def _get_boundary_elements_index(self):
        '''
        returns 
        index_list ::: list of int ::: indexes of elements located at a boundary
        '''
        index_list = [i for i,el_conn in enumerate(self.elements_intf_conn) if  len(el_conn) < 4]
        return index_list

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
            [face1, face2, face3, face4] = self._get_element_faces(element)
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
