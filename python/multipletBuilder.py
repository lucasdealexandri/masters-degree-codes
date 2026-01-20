import itertools
import numpy as np
from fractions import Fraction
from more_itertools import consecutive_groups
from typing import List, Set, Generator, Dict, Tuple

MW = 79.8244 # GeV. Value from MH // NumericalValue in FeynRules (SM.fr)
MH = 125 # GeV. Value from SM.fr

gw = 0.648397 # Weak coupling constant. Value from gw // NumericalValue in FeynRules (SM.fr)

def integer_partitions(n: int, max_val: int = None) -> Generator:
    """
    Find all integer partitions of the number n with values up to max_n
    Example: integer_partitions(5, 2) returns [[2, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]]
    """
    # Default value of max_n = n.
    if max_val == None: max_val = n
    if n == 0:
        yield []
        return 
    
    for i in range(min(max_val, n), 0, -1):
        for partition in integer_partitions(n - i, i):
            yield [i] + partition 


class MassEigenstate:
    # A mass eigenstate is defined by its mass and its charge. (In the future SU(3) indices will be added as well)
    # There is a third index, K, for disambiguation. This will be automatically assigned for particles with the same
    # mass and charge. The user must be careful not to enter the same particle twice.
    def __init__(self, mass: float, charge: float | Fraction, K: int = 0):
        self.label = f"Chi{charge}{K}"
        self.mass = mass
        self.charge = charge
        self.K = K
    
    def __repr__(self):
        return f"Mass eigenstate (m, q, K) = ({self.mass:.2f}, {self.charge}, {self.K})"
    
    def __str__(self): 
        return self.__repr__()
    
    def __eq__(self, other): 
        return self.label == other.label
        # return (self.mass == other.mass and self.charge == other.charge and self.K == other.K) 
    

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
        # We use mH = mW = 0 because our actual final state does not involve H and W, but rather quarks and fermions
        p1 = (mi**2-mj**2) / (2*mi)
        
        # Everything that is not U^{-1}.T.U (SU(2) group factor in mass basis) in the decay width for the charged decay
        # Γ = φ(mi, mj, mW) (U^{-1}.T.U)^2. phase_space_term is φ(mi, mj, mW).
        if self._charged_decay:
            # Calculated in FeynCalc (FeynCalc1To2FermionWDecay.nb)
            numerator = gw**2 * (mi-mj-MW)*(mi-mj+MW)*((mi+mj)**2 + 2*MW**2)*p1
            denominator = 16*np.pi * mi**2 * MW**2
            return numerator / denominator
        
        # In case this is a decay to the Higgs, 
        # Γ = φH(mi, mj, mW) (U^{-1}.Λ.U)^2, where Λ are the Yukawas times the Clebsh-Gordan coefficients
        else:
            # Calculated in FeynCalc (FeynCalc1To2FermionHDecay.nb)
            numerator = ((mi+mj)**2 - MH**2) * p1
            denominator = 8*np.pi * mi**2
            return numerator / denominator
        
    # BR mod phase space, i.e., 
    # its (U^{-1}.Λ.U)^2 / ΓTot for the neutral decay or (U^{-1}.T.U)^2 / ΓTot for the charged one.
    @property
    def modified_branching_ratio(self):
        return self.branching_ratio / self.phase_space_term
    
    def __eq__(self, other):
        # Two decays are the same if they involve the same particles and have the same branching ratios.
        # Ideally I'll add a __hash__ method later on to be able to check like {self.p1, self.p2} == {other.p1, other.p2}
        same_particles = (self.p1 == other.p1 and self.p2 == other.p2) or (self.p1 == other.p2 and self.p2 == other.p1)
        same_br = self.branching_ratio == other.branching_ratio
        
        return same_particles and same_br
    
    def __repr__(self):
        more_massive_particle = max(self.p1, self.p2, key=lambda p: p.mass)
        less_massive_particle = min(self.p1, self.p2, key=lambda p: p.mass)
        boson = "W" if self._charged_decay else "H"
        return f"{more_massive_particle.label} -> {less_massive_particle.label} + {boson}"


def mod_braching_fraction(decay: DecayChannel, all_decays: List[DecayChannel]):
        
    numerator = decay.modified_branching_ratio
    denominator = 0
    most_massive_particle = max([decay.p1, decay.p2], key=lambda p: p.mass)
    # If the most massive particle of the decay of interest is the same as
    # the most massive particle in other decays, then append the mBR of the other decay to the denominator.
    for d in all_decays:
        if most_massive_particle == max([d.p1, d.p2], key=lambda p: p.mass):
            denominator += d.modified_branching_ratio

    return numerator / denominator


def local_weight(decay: DecayChannel, all_decays: List[DecayChannel]):
    
    m1 = decay.p1.mass
    m2 = decay.p2.mass
    
    bf = mod_braching_fraction(decay, all_decays)
    return float(np.sqrt(bf * min(m1 / m2, m2 / m1))) 


class GaugeEigenstate:
    
    # The disambiguation index I is automatically assigned.
    # Initially, we assume to know nothing about a given gauge eigenstate. The quantum numbers will be inherited from
    # the multiplet they are contained.
    def __init__(self, 
                 isospin: int | Fraction = None, 
                 hypercharge: int | Fraction = None, 
                 I: int = 0,
                 isospin_projection: int | Fraction = None):
        
        self.isospin = isospin
        self.hypercharge = hypercharge
        self.I = I
        self.isospin_projection = isospin_projection
        self.charge = hypercharge + isospin_projection
        self.label = f"({isospin}, {hypercharge}, {I}, {isospin_projection})"
        
    def __repr__(self):
        # (j, y, I, m)
        return f"({self.isospin}, {self.hypercharge}, {self.I}, {self.isospin_projection})"
    
    def __str__(self): return self.__repr__()
    
    
class MultipletSolver:
    
    def __init__(self, particles: List[MassEigenstate], all_decays: List[DecayChannel]):
        self.particles = particles
        self.all_decays = all_decays
        self.charged_decays = [decay for decay in all_decays if decay._charged_decay]
        self.neutral_decays = [decay for decay in all_decays if not decay._charged_decay]
        self.all_charges = list(set([p.charge for p in particles]))
    
    def get_mass_eigenstate(self, label: str) -> MassEigenstate:
        return [p for p in self.particles if p.label == label][0]
        
    @property
    def all_partitions(self) -> List[List[int]]:
        """All integers partitions with values up to max_multiplet (the largest possible multiplet)"""
        # This is the length of the largest consecutive sublist of charges. Example:
        # if the charges were [-10, -9, -8, 5, 6, 7, 8], the largest consecutive sublist is [5, 6, 7, 8],
        # thus max_multiplet would be 4 (a quartet).
        max_multiplet = max([len(list(group)) for group in consecutive_groups(self.all_charges)])
        partitions = list(integer_partitions(len(self.particles), max_multiplet))
        # If there is any connection to the W boson (charged decay), the configuration of all singlets
        # is not reasonable, so it is removed.
        if any([decay._charged_decay for decay in self.all_decays]):
            partitions.pop()
        return partitions
        
    def construct_configuration(self, partition: List[int]) -> List[GaugeEigenstate]:
        """
        Based on the partition, constructs GaugeEigenstates.
        This is the configuration for one partition.
        """
        # I need a better way to think about how to fit charges to the multiplets...
        # So far, a naive method is implemented.
        gauge_basis = []
        charges = sorted([p.charge for p in self.particles], reverse=True)
        multiplets = []
        for n_multiplet in partition:
            I = 0
            # Starts from largest multiplet, places smallest charge on the bottom.
            j = Fraction(n_multiplet - 1, 2)
            # q = y + m => y = q - m. The smallest charge is at the bottom of the multiplet, where m = -j.
            y = charges[-1] + j
            charges.pop()
            # Update the desambiguation index I. For each different multiplet with
            # the same quantum numbers j and y, increase I by one.
            while (j, y, I) in multiplets:
                I += 1
            multiplets.append((j, y, I))
            for i in range(n_multiplet):
                m = i - j
                gauge_basis.append(GaugeEigenstate(j,y,I,m))
                
        return gauge_basis  
    
    @property
    def all_configurations(self) -> List[List[GaugeEigenstate]]:
        return [self.construct_configuration(partition) for partition in self.all_partitions]
    
    def assign_particles(self, normalize_scores: bool=True) -> List[Tuple[float, Dict[str, GaugeEigenstate]]]:
        """
        Assuming that each mass eigenstate has a predominant gauge eigenstate associated with it,
        assigns mass eigenstates to gauge eigenstates for every possible configuration,
        calculates the associated score of each configuration and returns the scores with the configurations.
        """
        all_possible_assignments = []
        for configuration in self.all_configurations:
            # What are all possible assignments for each configuration (list of multiplets)?
            possible_assignments = []
            
            # Separate in charges to avoid doing more permutations than necessary
            for charge in self.all_charges:
                particles = [particle for particle in self.particles if particle.charge == charge]
                permutations = itertools.permutations([g for g in configuration if g.charge == charge])
                
                # All permutations of ways to relate the mass eigenstates to the gauge eigenstates
                possible_assignment = [
                    {p.label: g for p,g in zip(particles, perm)} for perm in permutations]
                possible_assignments.append(possible_assignment)

            # Cartesian product of the assignments of different charges
            all_possible_assignments.append(
                [{k: v for d in combo for k, v in d.items()} for combo in itertools.product(*possible_assignments)])
        
         # Flatten the list
        all_possible_assignments = list(itertools.chain(*all_possible_assignments))
        
        # For each possible assignment, calculate the total score
        # and keep track of max score to normalize.
        scores = [self.calculate_total_score(assignment) for assignment in all_possible_assignments]
        max_score = max(scores) if normalize_scores else 1
        all_possible_assignments_ranked = [
            (score/max_score, assignment) for score, assignment in zip(scores, all_possible_assignments) if self.check_consistency(assignment)]
        
        # return all_possible_assignments
        return sorted(all_possible_assignments_ranked, key=lambda e: e[0], reverse=True)
    
    def check_consistency(self, assignment: Dict[str, GaugeEigenstate]) -> bool:
        # Check if a given assignment of particles to multiplets makes physical sense
        
        # If a singlet participates on a charged decay, it has to have a connection via Higgs to a multiplet.
        # So we check if there's any neutral decay to a component of a multiplet. If there isn't, the whole assignment is invalid.
        
        # if self.connections(assignment) == False: return False
        
        gauge_eigenstates = assignment.values()
        multiplets = sorted(gauge_eigenstates, key=lambda g: (g.isospin, g.hypercharge, g.I))
        multiplets = [list(g) for _, g in itertools.groupby(multiplets, key=lambda g: (g.isospin, g.hypercharge, g.I))]
        
        not_singlets = [m for m in multiplets if len(m) > 1]
        
        for mass_label, gauge_eigenstate in assignment.items():
            singlet = gauge_eigenstate.isospin == 0
            if singlet:
                # checks if there are charged decays involving it
                involved_in_charged_decay = any([mass_label in (decay.p1.label, decay.p2.label) for decay in self.charged_decays])
                
                # check if there are neutral decays involving it and a member of a multiplet
                involved_in_neutral_decay_with_multiplet = any(
                    [mass_label in (decay.p1.label, decay.p2.label) for decay in self.neutral_decays if any(
                        [g in multiplet for g in (assignment[decay.p1.label], assignment[decay.p2.label]) for multiplet in not_singlets])])
                
                if involved_in_charged_decay and not involved_in_neutral_decay_with_multiplet: return False
                
        return True
        
    def calculate_total_score(self, assignment: Dict[str, GaugeEigenstate]):
        """
        Locates particles in multiplets and calculates the score of their decays
        """
        gauge_eigenstates = assignment.values()
        
        # Locate particles gauge eigenstates that are in the same multiplet and separate them in lists
        multiplets = sorted(gauge_eigenstates, key=lambda g: (g.isospin, g.hypercharge, g.I))
        multiplets = [list(g) for _, g in itertools.groupby(multiplets, key=lambda g: (g.isospin, g.hypercharge, g.I))]
        
        # Now, for each multiplet, locate the particle that is assigned to each gauge eigenstate
        # and calculate the score of the decay
        total_score = 0
        for multiplet in multiplets:
            # Sort the multiplets by m so that charges go up by one always.
            multiplet = sorted(multiplet, key=lambda m: m.isospin_projection)
            mass_eigenstates = []
            for gauge_eigenstate in multiplet:
                # Find the mass eigenstate from the dictionary
                mass_label = [key for key, val in assignment.items() if val == gauge_eigenstate][0]
                # Get the MassEigenstate object
                mass_eigenstate = self.get_mass_eigenstate(mass_label)
                mass_eigenstates.append(mass_eigenstate)
            
            # Sum for each multiplet the adjacent-pair sum of the weight of decay channels 
            total_score += sum(
                self.calculate_weight(p1, p2) for p1, p2 in zip(mass_eigenstates, mass_eigenstates[1:]))
            
        return total_score
    
    def calculate_weight(self, p1: MassEigenstate, p2: MassEigenstate):
        relevant_decay = [decay for decay in self.all_decays if ((decay.p1 == p1 and decay.p2 == p2) or (decay.p1 == p2 and decay.p2 == p1))][0]
        return local_weight(relevant_decay, self.all_decays)
    
    def connections(self, assignment: Dict[str, GaugeEigenstate]) -> Dict[str, List[GaugeEigenstate]] | bool:
        """
        Gets an assignment (A dictionary relating mass eigenstate labels to gauge eigenstates)
        and returns, for each mass eigenstate, all the gauge eigenstates that constitute it.
        EXCEPT if the configuration is deemed invalid, then returns False.
        """
        
        # Begin with the mandatory connections
        connections = {mass_label: [gauge_eigenstate] for mass_label, gauge_eigenstate in assignment.items()}
        
        # Then, for each neutral decay, append to the connections
        for decay in self.neutral_decays:
            # If there's a nonzero branching ratio (a valid neutral decay) and the difference of isospins |j1 - j2| is not half,
            # this configuration is invalid and should be discarded.
            # if decay.branching_ratio != 0 and abs(
            #     assignment[decay.p1.label].isospin - assignment[decay.p2.label].isospin) != Fraction(1,2):
            #     return False 
            connections[decay.p1.label].append(assignment[decay.p2.label])
            connections[decay.p2.label].append(assignment[decay.p1.label])
            
        return connections
            
        

# class SU2Multiplet(list):
#     # An SU(2) Multiplet is a list of gauge eigenstates
#     def __init__(self, initial_multiplet: List[GaugeEigenstate] = None):
#         super().__init__()
    
#     # The isospin of an N-plet is given by 1/2 (N - 1)    
#     @property
#     def isospin(self): return Fraction(1,2)*(len(self) - 1)
    
#     # This likely won't stay for long
#     @property
#     def hypercharge(self): return self.initial_multiplet[0].hypercharge


if __name__ == "__main__":
    
    particles = [
        MassEigenstate(mass, charge, K) for mass, charge, K in zip(
            (794.3262972602188, 1200.303231104628, 1205.9769338444082, 1200.0, 1200.0),
            (0, 0, 0, 1, 1),
            (0, 1, 2, 1, 2)
    )]
    
    # Values from FeynRules considering MH = MW = 0.
    all_decays = [
        DecayChannel(p1, p2, br) for (p1, p2, br) in [
            (particles[0], particles[1], 0.928414),
            (particles[0], particles[2], 0.335743),
            (particles[1], particles[2], 0.0000449535),
            (particles[0], particles[2 + 1], 1.0),
            (particles[0], particles[2 + 2], 1.0),
            (particles[1], particles[2 + 1], -0.0547852),
            (particles[1], particles[2 + 2], -3.49735e-9),
            (particles[2], particles[2 + 1], -1.59078e-7),
            (particles[2], particles[2 + 2], -0.103109),
        ]
    ]
    
    solver = MultipletSolver(particles, all_decays)
    # print(solver.charged_decays)
    # print(solver.neutral_decays)
    # print(all_decays)
    # print(solver.all_configurations)
    # print(solver.construct_configuration([2,1,1,1]))
    # print(solver.check_consistency(1))
    # print(solver.all_configurations)
    
    # For the SDDM the configurations are duplicated since we have two doublets.
    for assignment in solver.assign_particles():
        print(assignment)
    print(len(solver.assign_particles()))
    # print(list(itertools.permutations(solver.all_configurations[0])))

    
    # for decay in all_decays: print(decay.modified_branching_ratio)
    # print()
    # for decay in all_decays: print(mod_braching_fraction(decay, all_decays))
    
    # Testing integer_partitions
    # for partition in integer_partitions(5, 2):
    #     print(partition)
    # for partition in integer_partitions(10, 7):
    #     print(partition)
