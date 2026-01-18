from typing import List
from fractions import Fraction


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
    

class NeutralDecay:
    
    def __init__(self, p1: MassEigenstate, p2: MassEigenstate, branching_ratio: float):
        pass
    
    
class ChargedDecay:
    
    def __init__(self, p1: MassEigenstate, p2: MassEigenstate, branching_ratio: float):
        pass


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