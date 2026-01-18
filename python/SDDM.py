import cmath
import numpy as np
from sympy import N
from typing import List
from datetime import date
from sympy.physics.quantum.cg import CG

fix_ident = lambda s: "\n".join([line.replace(" ", "", 8) for line in s.split("\n")]) 

numerical_cg = lambda j1, m1, j2, m2, j, m: N(CG(j1, m1, j2, m2, j, m).doit())


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
        # Notice the mass term involving ms has a 1/2 term, hence double the range.
        self.ms = self.rng.uniform(low = 1000.0, high = 4000.0) if ms == None else ms
        self.mu = self.rng.uniform(low = 500.0, high = 2000.0) if mu == None else mu
        
        # preference for yukawas in the range 0.01 - 1
        self.y1 = self.rng.uniform(low = 0.01, high = 1.0) if y1 == None else y1
        self.y2 = self.rng.uniform(low = 0.01, high = 1.0) if y2 == None else y2
        
        # Higgs' vev considering v/sqrt(2) as the convention
        self.vev = 246.221 # GeV
        
        self.neutral_mass_matrix = np.array(
            [
                [self.ms/2, -self.vev*self.y1, self.vev*self.y2],
                [-self.vev*self.y1, -self.mu, 0.0],
                [self.vev*self.y2, 0.0, self.mu]
                ])
         
        self.charged_mass_matrix = np.diag((-self.mu, self.mu))
        
        # Values from SM.fr using NumericalValue in Mathematica
        self.sw2 = 0.233699 # Sine of the Weinberg angle squared
        self.cW = np.sqrt(1 - self.sw2) # Cosine of the Weinberg angle
        self.g = 0.648397 # Weak coupling constant
        
    # The only multiplets that exist in the SDDM model with a given y (hypercharge), j (isospin), and i (degeneracy breaker)
    def allowed_multiplets(self, charge: int) -> List[tuple]:
        assert charge in [0, 1], "The only charges present are 0 and 1."
        options = {
            0: [(0, 0, 1), (1/2, 1/2, 1), (1/2, 1/2, 2)],
            1: [(1/2, 1/2, 1), (1/2, 1/2, 2)]
        }
        return options[charge]
    
    def yukawa(self, multiplet1, multiplet2):
        if multiplet1 == (1/2, 1/2, 1) and multiplet2 == (0, 0, 1): return self.y1
        elif multiplet1 == (1/2, 1/2, 2) and multiplet2 == (0, 0, 1): return self.y2
        else: return 0
    
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
    
    # Relates the bases like \chi^q_k = U^q_{k \alpha} \psi_\alpha, where \chi is in mass basis and \psi in gauge basis
    def change_of_basis(self, charge: int, k: int, multiplet: tuple):
        if multiplet not in self.allowed_multiplets(charge): return 0
        elif charge == 0:
            return self.change_of_basis_neutral()[k, self.mapping(0, multiplet)[1]]
        elif charge == 1:
            return np.array([[0,0,0],[0,1,0],[0,0,1]])[k, self.mapping(1, multiplet)[1]]
        else: return 0
        
    # Relates the bases like \psi_\alpha = (U^{-1})^q_{\alpha k} \chi^q_k
    def inverse_change_of_basis(self, multiplet: tuple, charge: int, k: int):
        if multiplet not in self.allowed_multiplets(charge): return 0
        elif charge == 0:
            return np.linalg.inv(self.change_of_basis_neutral())[self.mapping(0, multiplet)[1], k]
        elif charge == 1:
            return np.array([[0,0,0],[0,1,0],[0,0,1]])[self.mapping(1, multiplet)[1], k]
        else: return 0
       
    @property 
    def mass_of(self) -> dict:
        """Get the mass of a given (q, k) particle in mass basis"""
        return {particle: mass for particle, mass in zip([(0,0),(0,1),(0,2),(1,1),(1,2)], self.physical_masses)}
    
    # Make a .fr (FeynRules) file with the masses, couplings, and definitions calculated in this class
    def generate_fr(self) -> None:
        
        header = fix_ident(f"""
        (* This file was automatically generated from a script. *)
        
        M$ModelName = "Dirac SDDM"
        
        M$Information = {{
            Authors -> {{"Lucas Heck"}},
            Version -> "1.0",
            Date -> "{date.today().strftime("%d.%m.%Y")}",
            Institutions -> {{"Universidade Federal do ABC (UFABC)"}},
            Emails -> {{"lucas.heck@ufabc.edu.br"}}
        }};
        """)
        
        parameters = "\nM$Parameters = {"
        for name, value, description in zip(
            ["y1", "y2", "Ms", "Mc"], 
            [self.y1, self.y2, self.ms, self.mu], 
            ['"Coupling y1"', '"Coupling y2"', '"Singlet mass term"', '"Charged particles mass"']):
            
            parameters += fix_ident(f"""
            {name} == {{
                ParameterType -> External,
                Value -> {value},
                Description -> {description}
            }}{"," if name != "Mc" else ""}
            """)
            
        parameters += "\r};\n\n"
        
        classes = "M$ClassesDescription = {"
        # There are 5 physical fields (Mass eigenstates) and 3 unphysical ones (Gauge eigenstantes)
        physical_fields = [
            {"number": number, 
             "class_name": class_name, 
             "mass": mass, 
             "charge": charge} for number, class_name, mass, charge in zip(
                 [100, 101, 102, 111, 112], 
                 ["Chi00", "Chi01", "Chi02", "Chi11", "Chi12"],
                 self.physical_masses,
                 [0, 0, 0, 1, 1]
             )]
        
        for field in physical_fields:
            number, class_name, mass, charge = field.values()
            classes += fix_ident(f"""
            F[{number}] == {{
                ClassName -> {class_name},
                SelfConjugate -> False,
                Mass -> {{Mn{number-100:02d}, {mass}}},
                QuantumNumbers -> {{Q -> {charge}}}
            }},
            """)
            
        unphysical_fields = [
            {
            "number": number,
            "class_name": class_name,
            "doublet": doublet,
            "class_members": class_members,
            "hypercharge": hypercharge,
            "definitions": definitions
            } for number, class_name, doublet, class_members, hypercharge, definitions in zip(
                [1000, 1001, 1002],
                ["Psi0", "Psi1", "Psi2"],
                [False, True, True],
                [None, "{Psi11, Psi01}", "{Psi12, Psi02}"],
                [0, "1/2", "1/2"],
                [
                    f"{{Psi0[sp_] -> {" + ".join([f"{self.inverse_change_of_basis((0,0,1), 0, k)} Chi0{k}[sp]" for k in range(3)])}}}", 
                    f"""{{Psi1[sp_, 2] -> {" + ".join([f"{self.inverse_change_of_basis((1/2,1/2,1), 0, k)} Chi0{k}[sp]" for k in range(3)])}, 
                        Psi1[sp_, 1] -> Chi11[sp]}}""", 
                    f"""{{Psi2[sp_, 2] -> {" + ".join([f"{self.inverse_change_of_basis((1/2,1/2,2), 0, k)} Chi0{k}[sp]" for k in range(3)])}, 
                        Psi2[sp_, 1] -> Chi12[sp]}}"""
                ]
                )
        ]
        
        for field in unphysical_fields:
            number, class_name, doublet, class_members, hypercharge, definitions = field.values()
            doublet_stuff = fix_ident(f"""
                        ClassMembers -> {class_members},
                        Indices -> {{Index[SU2D]}},
                        FlavorIndex -> SU2D,""")
            classes += fix_ident(f"""
            F[{number}] == {{
                ClassName -> {class_name},{f"{doublet_stuff}" if doublet else ""}
                Unphysical -> True,
                SelfConjugate -> False,
                QuantumNumbers -> {{Y -> {hypercharge}}},
                Definitions -> {definitions}
            }}{"," if number != 1002 else ""}
            """)
        classes += "\r};"
        
        lag = "\n\nLKin = I Psi0bar . Ga[mu] . del[Psi0, mu] + I Psi1bar . Ga[mu] . DC[Psi1, mu] + I Psi2bar . Ga[mu] . DC[Psi2, mu];"
        lag += "\nLMass = -1/2 Ms Psi0bar[sp] . Psi0[sp] + Mc (Psi1bar[sp, ii] . Psi1[sp, ii] - Psi2bar[sp, ii] . Psi2[sp, ii]);"
        lag += "\nLY = y1 Phi[ii] (Psi1bar[sp, ii] . Psi0[sp]) + y2 Phi[ii] (Psi2bar[sp, ii] . Psi0[sp]) + HC[y1 Phi[ii] (Psi1bar[sp, ii] . Psi0[sp]) + y2 Phi[ii] (Psi2bar[sp, ii] . Psi0[sp])];"
        lag += "\n\nLBSM = LKin + LMass + LY;"
        
        return header + parameters + classes + lag
        
    @property
    def allowed_decays(self) -> dict:
        neutral_masses = self.physical_masses[:3]
        charged_masses = self.physical_masses[3]
        decays_from_chi00_W = [(1,k) for k in range(1,3) if neutral_masses[0] > charged_masses]
        decays_from_chi01_W = [(1,k) for k in range(1,3) if neutral_masses[1] > charged_masses]
        decays_from_chi02_W = [(1,k) for k in range(1,3) if neutral_masses[2] > charged_masses]
        decays_from_chi11_W = [(0,k) for k in range(3) if charged_masses > neutral_masses[k]]
        decays_from_chi12_W = [(0,k) for k in range(3) if charged_masses > neutral_masses[k]]
        decays_from_chi00_neutral = [(0, k) for k in range(3) if neutral_masses[0] > neutral_masses[k]]
        decays_from_chi01_neutral = [(0, k) for k in range(3) if neutral_masses[1] > neutral_masses[k]]
        decays_from_chi02_neutral = [(0, k) for k in range(3) if neutral_masses[2] > neutral_masses[k]]
        return {
            (0,0): decays_from_chi00_W + decays_from_chi00_neutral, 
            (0,1): decays_from_chi01_W + decays_from_chi01_neutral, 
            (0,2): decays_from_chi02_W + decays_from_chi02_neutral, 
            (1,1): decays_from_chi11_W, 
            (1,2): decays_from_chi12_W
            }
    
    # For the two body decay Xi -> Xj + W, the spin averaged |M|^2 has a mass term embedded.
    # This needs to be taken into consideration for BR and BF calculations.
    # Masses in GeV
    def mass_factor(self, m1: float, m2: float, third: str = "W"):
        
        mW = 80.37 # GeV
        mZ = 91.19 # GeV
        mH = 125 # GeV
        
        if third.lower() == "w":
            mM = mW
        elif third.lower() == "z":
            mM = mZ
        elif third.lower() == "h":
            mM = mH
        
        # q . p
        qp = 1/2 * (m1**2 + m2**2 - mM**2)
        # k . p
        kp = 1/2 * (m1**2 - m2**2 + mM**2)
        # k . q
        kq = 1/2 * (m1**2 - m2**2 - mM**2)
        
        if third.lower() in ("w", "z"):
            return (2*kp*kq) / mM ** 2 + (qp - 3*m1*m2)
        elif third.lower() == "h":
            return qp + m1*m2
    
    # p* times mass_factor
    def phase_space(self, M: float, m1: float, third: str = "W"):
        mW = 0 # The decay does not happen to the W, but to a quark/lepton final state in general
        # print(np.sqrt((M**2 - (m1 + mW)**2)*(M**2 - (m1 - mW)**2)) / (2*M))
        return np.sqrt((M**2 - (m1 + mW)**2)*(M**2 - (m1 - mW)**2)) / (2*M) * self.mass_factor(M, m1, third)
    
    def Z_matrix_element(self, charge: int, k1: int, k2: int):
        group_factor = lambda multiplet: multiplet[0]
        return self.g / self.cW * sum(
            group_factor(multiplet) * self.change_of_basis(charge, k1, multiplet) * self.change_of_basis(charge, k2, multiplet) 
            for multiplet in self.allowed_multiplets(charge)
            )
        
    def h_matrix_element(self, charge: int, k1: int, k2: int):
        def group_factor(multiplet1, multiplet2): 
            y1, j1, _ = multiplet1
            y2, j2, _ = multiplet2
            m1 = charge - y1
            m2 = m1 + 1/2
            return self.yukawa(multiplet1, multiplet2) * numerical_cg(j1, m1, j2, m2, 1/2, -1/2) * int(-0.05 < y2 - y1 + 1/2 < 0.05)
        return 1 / np.sqrt(2) * sum(
            group_factor(multiplet1, multiplet2) * self.change_of_basis(charge, k1, multiplet1) * self.change_of_basis(charge, k2, multiplet2) 
            for multiplet1 in self.allowed_multiplets(charge) for multiplet2 in self.allowed_multiplets(charge)
            )
    
    # Related to the element \chi^q_{k1} \chi^{q-1}_{k2} W+
    def Wp_matrix_element(self, charge: int, k1: int, k2: int):
        group_factor = lambda multiplet: np.sqrt(multiplet[0]*(multiplet[0] + 1) - (charge - multiplet[1])*(charge - multiplet[1] - 1))
        return self.g / np.sqrt(2) * sum(
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
        
    def W_decay_width(self, particle_in, particle_out):
        if particle_in[0] == particle_out[0]: return 0
        M = self.mass_of[particle_in]
        m1 = self.mass_of[particle_out]
        p_star = self.phase_space(M, m1, "W") / self.mass_factor(M, m1, "W")
        return p_star / (8*np.pi*M**2) * (self.Wm_matrix_element(0, particle_in[1], particle_out[1])**2 + self.Wp_matrix_element(1, particle_in[1], particle_out[1])**2)    
    
    def Z_decay_width(self, particle_in, particle_out):
        if particle_in[0] != particle_out[0]: return 0
        M = self.mass_of[particle_in]
        m1 = self.mass_of[particle_out]
        p_star = self.phase_space(M, m1, "Z") / self.mass_factor(M, m1, "Z")
        return p_star / (8*np.pi*M**2) * self.Z_matrix_element(particle_in[0], particle_in[1], particle_out[1])**2 
    
    def h_decay_width(self, particle_in, particle_out):
        if particle_in[0] != particle_out[0]: return 0
        M = self.mass_of[particle_in]
        m1 = self.mass_of[particle_out]
        p_star = self.phase_space(M, m1, "H") / self.mass_factor(M, m1, "H")
        return p_star / (8*np.pi*M**2) * self.h_matrix_element(particle_in[0], particle_in[1], particle_out[1])**2
    
    def total_decay_width(self, particle_in):
        result = sum(self.W_decay_width(particle_in, out) for out in self.allowed_decays[particle_in])
        result += sum(self.Z_decay_width(particle_in, out) for out in self.allowed_decays[particle_in])
        result += sum(self.h_decay_width(particle_in, out) for out in self.allowed_decays[particle_in])
        return result
    
    def BR(self, particle_in, particle_out, boson: str):
        total_width = self.total_decay_width(particle_in)
        if not any([particle_out in self.allowed_decays[particle_in]]): return 0
        if total_width == 0: return 0
        if boson.lower() == "w":
            return self.W_decay_width(particle_in, particle_out) / total_width
        elif boson.lower() == "z":
            return self.Z_decay_width(particle_in, particle_out) / total_width
        elif boson.lower() == "h":
            return self.h_decay_width(particle_in, particle_out) / total_width
        
    def BF(self, particle_in, particle_out, boson: str):
        if not any([particle_out in self.allowed_decays[particle_in]]): return 0
        if boson.lower() == "w":
            correct_width = self.W_decay_width
        elif boson.lower() == "z":
            correct_width = self.Z_decay_width
        elif boson.lower() == "h":
            correct_width = self.h_decay_width
            
        partial_width = sum(correct_width(particle_in, out) for out in self.allowed_decays[particle_in])
        if partial_width == 0: return 0
        return correct_width(particle_in, particle_out) / partial_width
    
    def tilde_BF(self, particle_in, particle_out, boson: str):
        if not any([particle_out in self.allowed_decays[particle_in]]): return 0
        if boson.lower() == "w":
            correct_width = self.W_decay_width
        elif boson.lower() == "z":
            correct_width = self.Z_decay_width
        elif boson.lower() == "h":
            correct_width = self.h_decay_width
        
        M = self.mass_of[particle_in]
        m1 = self.mass_of[particle_out]
        p_star = self.phase_space(M, m1, "H") / self.mass_factor(M, m1, "H")
        partial_width = sum(correct_width(particle_in, out) / (self.phase_space(M, self.mass_of[out], boson) / self.mass_factor(M, self.mass_of[out], boson)) for out in self.allowed_decays[particle_in])
        if partial_width == 0: return 0
        return (correct_width(particle_in, particle_out) / p_star) / partial_width
    
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
    
    sddm = SDDM(None, ms = 2*800, mu = 1200, y1 = 0.1, y2 = 0.2)
    print(sddm.physical_masses)
    # print(sddm.generate_fr())
    
    print(np.array([[sddm.BR(p1, p2, "W") for p1 in sddm.physical_particles] for p2 in sddm.physical_particles]))
    print()
    print(np.array([[sddm.BR(p1, p2, "H") for p1 in sddm.physical_particles] for p2 in sddm.physical_particles]))
    
    # print((sddm.BFW - sddm.tilde_BFW) / sddm.BFW)
    print(sddm.tilde_BFW)
    # print(sddm.allowed_decays)
    # string = f"\\mathrm{{BR}}(\\chi^1_1 \\to \\chi^0_0 + W^+) = {sddm.BR((1,1), (0,0), "W"):.6f} && "
    # string += " && ".join([f"\\mathrm{{BR}}(\\chi^0_{k} \\to \\chi^1_1 + W^+) = {sddm.BR((0,k), (1,1), "W"):.6f}" for k in range(1,3)])
    # string += "\\\\ \n"
    # string += f"\\mathrm{{BR}}(\\chi^1_2 \\to \\chi^0_0 + W^+) = {sddm.BR((1,2), (0,0), "W"):.6f} && "
    # string +=  " && ".join([f"\\mathrm{{BR}}(\\chi^0_{k} \\to \\chi^1_2 + W^+) = {sddm.BR((0,k), (1,2), "W"):.6f}" for k in range(1,3)])
    # BFtilde = f"\\widetilde{{\\mathrm{{BF}}}}(\\chi^1_1 \\to \\chi^0_0 + W^+) = {sddm.tilde_BF((1,1), (0,0), "W"):.6f}, && "
    # BFtilde += ", && ".join([f"\\widetilde{{\\mathrm{{BF}}}}(\\chi^0_{k} \\to \\chi^1_1 + W^-) = {sddm.tilde_BF((0,k), (1,1), "W"):.6f}" for k in range(1,3)])
    # BFtilde += "\\\\ \n"
    # BFtilde += f"\\widetilde{{\\mathrm{{BF}}}}(\\chi^1_2 \\to \\chi^0_0 + W^+) = {sddm.tilde_BF((1,2), (0,0), "W"):.6f}, && "
    # BFtilde +=  ", && ".join([f"\\widetilde{{\\mathrm{{BF}}}}(\\chi^0_{k} \\to \\chi^1_2 + W^-) = {sddm.tilde_BF((0,k), (1,2), "W"):.6f}" for k in range(1,3)])
    
    # print(BFtilde)
    # print(f"\\mathrm{{BR}}(\\chi^1_1 \\to \\chi^0_0 + W^+) = {sddm.BR((1,1), (0,0), "W"):.7f} && \\mathrm{{BR}}(\\chi^0_1 \\to \\chi^1_1 + W^-) = {sddm.BR((0,1), (1,1), "W"):.7f} && \\mathrm{{BR}}(\\chi^0_2 \\to \\chi^1_1 + W^-) = {sddm.BR((0,2), (1,1), "W"):.7f}")
    # print(f"\\mathrm{{BR}}(\\chi^1_2 \\to \\chi^0_0 + W^+) = {sddm.BR((1,2), (0,0), "W"):.7f} && \\mathrm{{BR}}(\\chi^0_1 \\to \\chi^1_2 + W^-) = {sddm.BR((0,1), (1,2), "W"):.7f} && \\mathrm{{BR}}(\\chi^0_2 \\to \\chi^1_2 + W^-) = {sddm.BR((0,2), (1,2), "W"):.7f}")
    # print()
    # print(f"BF(11 -> 00 + W+) = {sddm.BF((1,1), (0,0), "W"):.7f}\t BF(01 -> 11 + W-) = {sddm.BF((0,1), (1,1), "W"):.7f}\t BF(02 -> 11 + W-) = {sddm.BF((0,2), (1,1), "W"):.7f}")
    # print(f"BF(12 -> 00 + W+) = {sddm.BF((1,2), (0,0), "W"):.7f}\t BF(01 -> 12 + W-) = {sddm.BF((0,1), (1,2), "W"):.7f}\t BF(02 -> 12 + W-) = {sddm.BF((0,2), (1,2), "W"):.7f}")
    # print()
    # print(f"~BF(11 -> 00 + W+) = {sddm.tilde_BF((1,1), (0,0), "W"):.7f}\t ~BF(01 -> 11 + W-) = {sddm.tilde_BF((0,1), (1,1), "W"):.7f}\t ~BF(02 -> 11 + W-) = {sddm.tilde_BF((0,2), (1,1), "W"):.7f}")
    # print(f"~BF(12 -> 00 + W+) = {sddm.tilde_BF((1,2), (0,0), "W"):.7f}\t ~BF(01 -> 12 + W-) = {sddm.tilde_BF((0,1), (1,2), "W"):.7f}\t ~BF(02 -> 12 + W-) = {sddm.tilde_BF((0,2), (1,2), "W"):.7f}")
    # print(sddm.tilde_BFW)
    # print(sddm.tilde_BFW)