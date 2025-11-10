import numpy as np
from typing import List

class SDDM:
    
    def __init__(self, 
                 seed: int = None,
                 ms: float = None, 
                 mu: float = None, 
                 y1: float = None, 
                 y2: float = None):
        
        self.seed = seed
        
        self.rng = np.random.default_rng(seed = seed)
        
        # preference for masses in the range of 500 - 2000 GeV
        self.ms = self.rng.uniform(low = 500.0, high = 2000.0) if ms == None else ms
        self.mu = self.rng.uniform(low = 500.0, high = 2000.0) if mu == None else mu
        
        # preference for yukawas in the range 0.01 - 1
        self.y1 = self.rng.uniform(low = 0.01, high = 1.0) if y1 == None else y1
        self.y2 = self.rng.uniform(low = 0.01, high = 1.0) if y2 == None else y2
        
        # Higgs' vev considering v/sqrt(2) as the convention
        self.vev = 246.22 # GeV
        
        self.neutral_mass_matrix = np.array(
            [
                [self.ms, -self.vev*self.y1, self.vev*self.y2],
                [-self.vev*self.y1, -self.mu, 0.0],
                [self.vev*self.y2, 0.0, self.mu]
                ])
         
        self.charged_mass_matrix = np.diag((-self.mu, self.mu))
        
    # The only multiplets that exist in the SDDM model with a given y (hypercharge), j (isospin), and i (degeneracy breaker)
    def allowed_multiplets(self, charge: int) -> List[tuple]:
        assert charge in [0, 1], "The only charges present are 0 and 1."
        options = {
            0: [(0, 0, 1), (1/2, 1/2, 1), (1/2, 1/2, 2)],
            1: [(1/2, 1/2, 1), (1/2, 1/2, 2)]
        }
        return options[charge]
    
    # Notation: (q, k), where q is charge and k is index
    @property
    def physical_particles(self):
        return ((0,0),(0,1),(0,2),(1,1),(1,2))
        
    # Mapping between the gauge eigenstates basis and the mass basis. The tuple (y, j, i) gets converted to (q, k)
    def mapping(self, charge: int, multiplet: tuple):
        options = {
            (0, self.allowed_multiplets(0)[0]): (0, 0),
            (0, self.allowed_multiplets(0)[1]): (0, 1),
            (0, self.allowed_multiplets(0)[2]): (0, 2),
            (1, self.allowed_multiplets(1)[0]): (1, 1),
            (1, self.allowed_multiplets(1)[1]): (1, 2)
                }
        return options[(charge, multiplet)]
    
    # Note: this returns the eigenvectors as the row vectors (the i-th eigenvector is given by eigenvectors[i,:])
    # It changes from gauge to mass basis and is sorted such that it is a perturbed identity matrix
    def change_of_basis_neutral(self):
        _, eigenvectors = np.linalg.eig(self.neutral_mass_matrix)
        permutation = np.argsort(np.argmax(np.abs(eigenvectors), axis=0)) # ensure that the matrix is as close as possible to the identity matrix
        return (eigenvectors[:, permutation]).T
    
    @property
    def physical_masses(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.neutral_mass_matrix)
        permutation = np.argsort(np.argmax(np.abs(eigenvectors), axis=0))
        eigenvalues = eigenvalues[permutation]
        return np.append(
            np.abs(eigenvalues), 
            np.array([self.mu, self.mu])
            )
    
    # Relates the basis like \chi^q_k = U^q_{k \alpha} \psi_\alpha, where \chi is in mass basis and \psi in gauge basis
    def change_of_basis(self, charge: int, k: int, multiplet: List[tuple]):
        if multiplet not in self.allowed_multiplets(charge): return 0
        elif charge == 0:
            return self.change_of_basis_neutral()[k, self.mapping(0, multiplet)[1]]
        elif charge == 1:
            return np.array([[0,0,0],[0,1,0],[0,0,1]])[k, self.mapping(1, multiplet)[1]]
        else: return 0
    
    # For the two body decay Xi -> Xj + W, the spin averaged |M|^2 has a mass term embedded.
    # This needs to be taken into consideration for BR and BF calculations.
    # Masses in GeV
    def mass_factor(self, m1: float, m2: float):
        mW = 80.37 # GeV
        # q . p
        qp = 1/2 * (m1**2 + m2**2 - mW**2)
        # k . p
        kp = 1/2 * (m1**2 - m2**2 + mW**2)
        
        return (qp + 3*m1*m2 + 2*kp**2 / mW**2)
    
    # p* times mass_factor
    def phase_space(self, M: float, m1: float):
        mW = 80.37 # GeV
        print((M**2 - (m1 + mW)**2)*(M**2 - (m1 - mW)**2))
        return 1 / (2*M) * np.sqrt((M**2 - (m1 + mW)**2)*(M**2 - (m1 - mW)**2)) * self.mass_factor(M, m1)
    
    # Related to the element \chi^q_{k1} \chi^{q-1}_{k2} W+
    def Wp_matrix_element(self, charge: int, k1: int, k2: int):
        group_factor = lambda multiplet: np.sqrt(multiplet[0]*(multiplet[0] + 1) - (charge - multiplet[1])*(charge - multiplet[1] - 1))
        return sum(
            group_factor(multiplet) * self.change_of_basis(charge, k1, multiplet) * self.change_of_basis(charge - 1, k2, multiplet) 
            for multiplet in self.allowed_multiplets(charge)
            )
        
    # Related to the element \chi^q_{k1} \chi^{q+1}_{k2} W-
    def Wm_matrix_element(self, charge: int, k1: int, k2: int):
        group_factor = lambda multiplet: np.sqrt(multiplet[0]*(multiplet[0] + 1) - (charge - multiplet[1])*(charge - multiplet[1] + 1))
        return sum(
            group_factor(multiplet) * self.change_of_basis(charge, k1, multiplet) * self.change_of_basis(charge + 1, k2, multiplet) 
            for multiplet in self.allowed_multiplets(charge)
            )
        
    # Measures the branching fraction from physical particle (q, k1) to another (q - 1, k2) and a W+
    # For the SDDM, q can only be 1.
    def BFWp(self, charge: int, k1: int, k2: int):
        m1 = self.physical_masses[2+k1]
        m2 = self.physical_masses[k2]
        numerator = self.Wp_matrix_element(charge, k1, k2)**2 * self.phase_space(m1, m2) if m1 > m2 else 0
        denominator = sum(self.Wp_matrix_element(charge, k1, k)**2 * self.phase_space(m1, self.physical_masses[k]) if m1 > self.physical_masses[k] else 0 for k in range(3))
        if denominator == 0: return 0
        return numerator / denominator
    
    # Measures the branching fraction from physical particle (q, k1) to another (q + 1, k2) and a W-
    # For the SDDM, q can only be 0.
    def BFWm(self, charge: int, k1: int, k2: int):
        m1 = self.physical_masses[k1]
        m2 = self.physical_masses[2 + k2]
        numerator = self.Wm_matrix_element(charge, k1, k2)**2 * self.phase_space(m1, m2) if m1 > m2 else 0
        denominator = sum(self.Wm_matrix_element(charge, k1, k)**2 * self.phase_space(m1, self.physical_masses[2+k]) if m1 > self.physical_masses[2+k] else 0 for k in range(1,3))
        if denominator == 0: return 0
        return numerator / denominator
        
    # Measures the modified branching fraction from physical particle (q, k1) to another (q - 1, k2) and a W+
    # For the SDDM, q can only be 1.
    def tilde_BFWp(self, charge: int, k1: int, k2: int):
        m1 = self.physical_masses[2+k1]
        m2 = self.physical_masses[k2]
        numerator = self.Wp_matrix_element(charge, k1, k2)**2 * self.mass_factor(m1, m2) if m1 > m2 else 0
        denominator = sum(self.Wp_matrix_element(charge, k1, k)**2 * self.mass_factor(m1, self.physical_masses[k]) if m1 > self.physical_masses[k] else 0 for k in range(3))
        if denominator == 0: return 0
        return numerator / denominator
    
    # Measures the modified branching fraction from physical particle (q, k1) to another (q + 1, k2) and a W-
    # For the SDDM, q can only be 0.
    def tilde_BFWm(self, charge: int, k1: int, k2: int):
        m1 = self.physical_masses[k1]
        m2 = self.physical_masses[2 + k2]
        numerator = self.Wm_matrix_element(charge, k1, k2)**2 * self.mass_factor(m1, m2) if m1 > m2 else 0
        denominator = sum(self.Wm_matrix_element(charge, k1, k)**2 * self.mass_factor(m1, self.physical_masses[2+k]) if m1 > self.physical_masses[2+k] else 0 for k in range(1,3))
        if denominator == 0: return 0
        return numerator / denominator
    
    @property
    def BFW(self):
        return np.array([[self.BFWp(1, k2, k1) + self.BFWm(0, k1, k2) for k1 in range(3)] for k2 in range(1,3)])
    
    @property
    def tilde_BFW(self):
        return np.array([[self.tilde_BFWp(1, k2, k1) + self.tilde_BFWm(0, k1, k2) for k1 in range(3)] for k2 in range(1,3)])
        
    
if __name__ == "__main__":
    
    sddm = SDDM()
    print(sddm.physical_masses)
    print(sddm.BFW)
    print(sddm.tilde_BFW)
