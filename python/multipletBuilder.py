import numpy as np
from typing import List
from fractions import Fraction

MW = 79.8244 # GeV. Value from MH // NumericalValue in FeynRules (SM.fr)
MH = 125 # GeV. Value from SM.fr
gw = 0.648397 # Weak coupling constant. Value from gw // NumericalValue in FeynRules (SM.fr)

class MassEigenstate:
    # A mass eigenstate is defined by its mass and its charge. (In the future SU(3) indices will be added as well)
    # There is a third index, K, for disambiguation. This will be automatically assigned for particles with the same
    # mass and charge. The user must be careful not to enter the same particle twice.
    def __init__(self, mass: float, charge: float | Fraction, K: int = 0):
        self.mass = mass
        self.charge = charge
        self.K = K
    
    def __repr__(self):
        return f"Mass eigenstate (m, q, K) = ({self.mass:.4g}, {self.charge}, {self.K})"
    
    def __str__(self): return self.__repr__()
    

WBOSON = MassEigenstate(MW, 1)
HIGGS = MassEigenstate(MH, 0)


class DecayChannel:
    
    def __init__(self, p1: MassEigenstate, p2: MassEigenstate, branching_ratio: float):
        self.p1 = p1
        self.p2 = p2
        self.branching_ratio = branching_ratio
        assert int(abs(p1.charge - p2.charge)) in (0,1), "Invalid charge difference between particles 1 and 2."
        # We are only considering decays to H and W. Charged decay => W, Neutral => H.
        self._charged_decay = int(abs(p1.charge - p2.charge)) == 1
        self.boson = WBOSON if self._charged_decay else HIGGS
    
    @property  
    def phase_space_term(self):
        
        mi = max(self.p1.mass, self.p2.mass)
        mj = min(self.p1.mass, self.p2.mass)
        
        # Magnitude of 3 momentum https://pdg.lbl.gov/2017/reviews/rpp2017-rev-kinematics.pdf
        p1W = np.sqrt((mi**2-(mj-MW)**2)*(mi**2-(mj+MW)**2)) / (2*mi)
        p1H = np.sqrt((mi**2-(mj-MH)**2)*(mi**2-(mj+MH)**2)) / (2*mi)
        
        # Everything that is not U^{-1}.T.U (SU(2) group factor in mass basis) in the decay width for the charged decay
        # Γ = φ(mi, mj, mW) (U^{-1}.T.U)^2. phase_space_term is φ(mi, mj, mW).
        if self._charged_decay:
            # Calculated in FeynCalc (FeynCalc1To2FermionWDecay.nb)
            numerator = gw**2 * (mi-mj-MW)*(mi-mj+MW)*((mi+mj)**2 + 2*MW**2)*p1W
            denominator = 8*np.pi * mi**2 * MW**2
            return numerator / denominator
        
        # In case this is a decay to the Higgs, 
        # Γ = φH(mi, mj, mW) (U^{-1}.Λ.U)^2, where Λ are the Yukawas times the Clebsh-Gordan coefficients
        else:
            # Calculated in FeynCalc (FeynCalc1To2FermionHDecay.nb)
            numerator = ((mi+mj)**2 - MH**2) * p1H
            denominator = 8*np.pi * mi**2
            return numerator / denominator
        
    # BR mod phase space, i.e., 
    # its (U^{-1}.Λ.U)^2 / ΓTot for the neutral decay or (U^{-1}.T.U)^2 / ΓTot for the charged one.
    @property
    def modified_branching_ratio(self):
        return self.branching_ratio / self.phase_space_term


class FlavorEigenstate:
    
    # The disambiguation index I is automatically assigned.
    # Initially, we assume to know nothing about a given flavor eigenstate. The quantum numbers will be inherited from
    # the multiplet they are contained.
    def __init__(self, 
                 isospin: int | Fraction = None, 
                 hypercharge: int | Fraction = None, 
                 isospin_projection: int | Fraction = None,
                 I: int = 0):
        
        self.isospin = isospin
        self.hypercharge = hypercharge
        self.isospin_projection = isospin_projection
        self.I = I
        
    def __repr__(self):
        return f"Flavor eigenstate (j, y, m, I) = ({self.isospin}, {self.hypercharge}, {self.isospin_projection}, {self.I})"
    
    def __str__(self): return self.__repr__()


class SU2Multiplet(list):
    # An SU(2) Multiplet is a list of flavor eigenstates
    def __init__(self, initial_multiplet: List[FlavorEigenstate] = None):
        super().__init__()
    
    # The isospin of an N-plet is given by 1/2 (N - 1)    
    @property
    def isospin(self): return Fraction(1,2)*(len(self) - 1)
    
    # This likely won't stay for long
    @property
    def hypercharge(self): return self.initial_multiplet[0].hypercharge
    
    
class Configuration(list):
    # A configuration is a lis
    def __init__(self, initial_configuration: List[SU2Multiplet] = None):
        super().__init_()


if __name__ == "__main__":
    
    print(MassEigenstate(12.312312, 1))