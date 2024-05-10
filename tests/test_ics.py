import pytest
import sys as sys 
sys.path.append('.')
from solver.incompressible import IncompressibleSolver
from meshe.mesh import *

@pytest.fixture()
def solver_fixture(): 
    solver = IncompressibleSolver()
    return solver

@pytest.fixture()
def simple_mesh_fixture():
    dx = 1
    n_elem = 3
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    return mesh 

@pytest.fixture()
def mesh_fixture():
    dx = 1
    n_elem = 21
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    #
    velocity = 1. 
    arr_tmp = np.zeros((n_elem,3))
    arr_tmp[:,0] = velocity * 1. 
    mesh.elements_data['velocity'] =  arr_tmp
    n_bndf = np.size(mesh.bndfaces,0)
    arr_tmp = np.zeros((n_bndf,3))
    arr_tmp[:,0] = velocity * 1. 
    arr_tmp[0,0] = 10
    mesh.bndfaces_data['velocity'] =    arr_tmp 
    return mesh 

def test_set_bnd_velocity(simple_mesh_fixture, solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(simple_mesh_fixture)
    solver_fixture.update_boundary_velocity(simple_mesh_fixture, 
                                            boundary_conditions)
    print(simple_mesh_fixture.bndfaces_data[solver_fixture.velocity_data])
    bnd_velocity = simple_mesh_fixture.bndfaces_data[solver_fixture.velocity_data]
    expected = np.array([[1, 0, 0],
                         [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                         [0, 0, 0]])
    assertion = np.all(bnd_velocity == expected) 
    assert assertion 
    
def test_setbc(solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,2,3])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    assertion = False
    assert assertion 

def test_advance_velocity(mesh_fixture, solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(mesh_fixture)
    density_value = 1000.
    dyna_visco = 1e-3
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, density_value, dyna_visco)
    n_ite = 1
    for _ in range(n_ite):
        print(_)
        solver_fixture.advance_velocity(mesh_fixture)
        solver_fixture.update_boundary_velocity(mesh_fixture, 
                                                boundary_conditions)
    print(mesh_fixture.elements_data[solver_fixture.velocity_data])
    print(np.mean(mesh_fixture.elements_data[solver_fixture.velocity_data],axis = 0))
    assertion = False
    assert assertion 
    
def test_pressure_lapl(mesh_fixture, solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(mesh_fixture)
    solver_fixture._set_pl_operators(mesh_fixture)
    print(solver_fixture.mat_pressure)
    print(solver_fixture.rhs_pressure)
    assertion = False
    assert assertion 

def test_velocity_divergence(mesh_fixture, solver_fixture):
    density = 1000 
    dyna_visco = 1e-3
    dt = 1.
    solver_fixture.set_constant_density(mesh_fixture, density)
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, density, dyna_visco)
    solver_fixture._update_velocity_div(mesh_fixture, deltat = dt)
    udiv = mesh_fixture.elements_data['div_velocity']
    assertion = False 
    assert assertion 

def test_calc_pressure(mesh_fixture, solver_fixture):
    density = 1000 
    dyna_visco = 1e-3
    deltat = 1
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([0.001,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    #
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(mesh_fixture)
    solver_fixture.set_constant_density(mesh_fixture, density)
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, density, dyna_visco)
    # Solve Pressure 
    solver_fixture.update_boundary_velocity(mesh_fixture, boundary_conditions)
    solver_fixture._update_velocity_div(mesh_fixture, deltat = deltat )
    solver_fixture._set_pl_operators(mesh_fixture)
    rhs = mesh_fixture.elements_data['div_velocity']
    mat = solver_fixture.mat_pressure
    expl = solver_fixture.rhs_pressure
    mesh_fixture.elements_data['pressure'] = np.linalg.solve(mat,rhs+expl)
    print(mat)
    print(expl)
    print(rhs)
    print(mesh_fixture.elements_data['pressure'])
    # Velocity correction 
    solver_fixture.update_boundary_pressure(mesh_fixture, boundary_conditions)
    solver_fixture.gradop(mesh_fixture)
    oo_rho = 1./mesh_fixture.elements_data['rho']
    corrector = -deltat*np.multiply(oo_rho,mesh_fixture.elements_data['grad_pressure'])
    mesh_fixture.elements_data['velocity'] = mesh_fixture.elements_data['velocity'] + corrector
    print(mesh_fixture.elements_data['grad_pressure'])
    print(mesh_fixture.elements_data['velocity'] )
    #
    for _ in range(10) :
        # Advance Velocity
        solver_fixture.advance_velocity(mesh_fixture)
        # Solve Pressure
        deltat = solver_fixture.timeop.dt
        solver_fixture.update_boundary_velocity(mesh_fixture, boundary_conditions)
        solver_fixture._update_velocity_div(mesh_fixture, deltat = deltat )
        solver_fixture._set_pl_operators(mesh_fixture)
        rhs = mesh_fixture.elements_data['div_velocity']
        mat = solver_fixture.mat_pressure
        expl = solver_fixture.rhs_pressure
        mesh_fixture.elements_data['pressure'] = np.linalg.solve(mat,rhs+expl)
        # Velocity correction 
        solver_fixture.update_boundary_pressure(mesh_fixture, boundary_conditions)
        solver_fixture.gradop(mesh_fixture)
        oo_rho = 1./mesh_fixture.elements_data['rho']
        corrector = -deltat*np.multiply(oo_rho,mesh_fixture.elements_data['grad_pressure'])
        mesh_fixture.elements_data['velocity'] = mesh_fixture.elements_data['velocity'] + corrector
    print(mesh_fixture.elements_data['pressure'])
    print(mesh_fixture.elements_data['velocity'] )

    
    
    
    assertion = False 
    assert assertion 