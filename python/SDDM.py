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
        
    # Measures the modified branching ratio from physical particle (q, k1) to another (q - 1, k2) and a W+
    # For the SDDM, q can only be 1.
    def tilde_BRWp(self, charge: int, k1: int, k2: int):
        numerator = self.Wp_matrix_element(charge, k1, k2)**2 if self.physical_masses[2+k1] > self.physical_masses[k2] else 0
        denominator = sum(self.Wp_matrix_element(charge, k1, k)**2 if self.physical_masses[2+k1] > self.physical_masses[k] else 0 for k in range(3))
        if denominator == 0: return 0
        return numerator / denominator
    
    # Measures the modified branching ratio from physical particle (q, k1) to another (q + 1, k2) and a W-
    # For the SDDM, q can only be 1.
    def tilde_BRWm(self, charge: int, k1: int, k2: int):
        numerator = self.Wm_matrix_element(charge, k1, k2)**2
        kinematically_possible = int(self.physical_masses[k1] > self.physical_masses[2 + k2])
        numerator *= kinematically_possible
        denominator = sum(self.Wm_matrix_element(charge, k1, k)**2 if self.physical_masses[k1] > self.physical_masses[2+k] else 0 for k in range(1,3))
        if denominator == 0: return 0
        return numerator / denominator
    
    @property
    def tilde_BRW(self):
        return np.array([[self.tilde_BRWp(1, k2, k1) + self.tilde_BRWm(0, k1, k2) for k1 in range(3)] for k2 in range(1,3)])
    
    # Metric for determining probability of \chi^{q}_{k1} being on top of \chi^{q-1}_{k2}
    # in an SU(2) multiplet
    def upstairs_metric(self, charge: int, k1: int, k2: int):
        neutral_mass = self.physical_masses[k2]
        charged_mass = self.physical_masses[2 + k1]
        delta_m = min(neutral_mass / charged_mass, charged_mass / neutral_mass)
        return np.sqrt(self.tilde_BRWp(charge, k1, k2) * delta_m)
    
    # Metric for determining probability of \chi^{q}_{k1} being below of \chi^{q+1}_{k2}
    # in an SU(2) multiplet
    def downstairs_metric(self, charge: int, k1: int, k2: int):
        neutral_mass = self.physical_masses[k1]
        charged_mass = self.physical_masses[2 + k2]
        delta_m = min(neutral_mass / charged_mass, charged_mass / neutral_mass)
        return np.sqrt(self.tilde_BRWm(charge, k1, k2) * delta_m)
    
    def multiplet_metric(self, particle1: List[tuple], particle2: List[tuple]):
        particles = [particle1, particle2]
        
        assert any(particle[0] == 0 for particle in particles), "No neutral particle found"
        assert any(particle[0] == 1 for particle in particles), "No charged particle found"
        
        neutral_particle = particle1 if particle1[0] == 0 else particle2
        charged_particle = particle1 if particle1[0] == 1 else particle2
        
        return self.upstairs_metric(1,neutral_particle[1],charged_particle[1]) + self.downstairs_metric(0,neutral_particle[1],charged_particle[1])
    
    def cluster_doublets(self):
        partners = []
        neutral_particles = self.physical_particles[:3]
        charged_particles = self.physical_particles[3:]
        
        for particle in charged_particles:
            distances = [self.multiplet_metric(particle, neutral_particle) for neutral_particle in neutral_particles]
            candidate = np.argmax(distances)
            partners.append((particle, neutral_particles[candidate]))    
            
        return partners
        
    
if __name__ == "__main__":
    
    sddm = SDDM()

    # print(sddm.allowed_multiplets(1))
    # print(sddm.mapping(0,(0,0,1)))
    # print(f"ms = {sddm.ms}, mu = {sddm.mu}, y1 = {sddm.y1}, y2 = {sddm.y2}")
    print(sddm.physical_masses)
    # print(sddm.change_of_basis_neutral())
    # print(sddm.change_of_basis(0,0,(0,0,1)))
    # print("k1 refers to charge = 1, k2 to charge = 0")
    # print(np.array([[f"(1, {k1}) -> (0, {k2})" for k1 in range(1, 3)] for k2 in range(3)]))
    # # print(np.array([[f"{sddm.physical_masses[k1]},{sddm.physical_masses[2+k2]}" for k1 in range(1, 3)] for k2 in range(3)]))
    # print(np.array([[sddm.tilde_BRWp(1, k1, k2) for k1 in range(1,3)] for k2 in range(3)]))
    # print("k1 refers to charge = 0, k2 to charge = 1")
    # print(np.array([[f"(0, {k1}) -> (1, {k2})" for k1 in range(3)] for k2 in range(1,3)]))
    # print(np.array([[sddm.tilde_BRWm(0, k1, k2) for k1 in range(3)] for k2 in range(1,3)]))
    # print(np.array([[sddm.tilde_BRWp(1, k2, k1) for k1 in range(3)] for k2 in range(1,3)]))
    print(np.array([[sddm.tilde_BRWp(1, k2, k1) + sddm.tilde_BRWm(0, k1, k2) for k1 in range(3)] for k2 in range(1,3)]))
    # print(np.array([[sddm.upstairs_metric(1, k2, k1) + sddm.downstairs_metric(0, k1, k2) for k1 in range(3)] for k2 in range(1,3)]))
    # print("upstairs metric")
    # print(np.array([[sddm.upstairs_metric(1, k1, k2) for k1 in range(3)] for k2 in range(3)]))
    # print("downstairs metric")
    # print(np.array([[sddm.downstairs_metric(0, k2, k1) for k1 in range(3)] for k2 in range(3)]))
    # print("combined metric")
    # print(np.array([[sddm.upstairs_metric(1, k1, k2) + sddm.downstairs_metric(0, k2, k1) for k1 in range(3)] for k2 in range(3)]))
    # print(sddm.cluster_doublets())