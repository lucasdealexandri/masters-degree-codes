import itertools
from typing import List, Set, Tuple, Dict
from fractions import Fraction
# We use UserDict instead of inheriting from dict for more predictable behavior
# See, e.g., https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
from collections import UserDict, defaultdict


class Particle:
    
    def __init__(self, label: str, spin: int | Fraction, color: int, charge: int | Fraction, mass: float):
        self.label = label
        self.spin = spin
        self.color = color
        self.charge = charge
        self.mass = mass
        
        self.unbroken_qns = (spin, color, charge)
        
    
    def __eq__(self, other):
        same_label = self.label == other.label
        same_spin = self.spin == other.spin
        same_color = self.color == other.color
        same_charge = self.charge == other.charge
        # Maybe be a little tolerant on slightly different masses?
        same_mass = self.mass == other.mass
        
        # dont use mass check for now.
        return same_label and same_spin and same_color and same_charge
    
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
        
    def __hash__(self):
        return hash((self.label, self.spin, self.color, self.charge))
        
    
    def __repr__(self):
        return self.label
    
    
    def __str__(self):
        return self.__repr__()
    
    
class DecayChannel:
    
    def __init__(self, p1: Particle, p2: Particle):
        self.p1 = p1
        self.p2 = p2
        
        self.most_massive = max(p1, p2, key = lambda p: p.mass)
        self.least_massive = min(p1, p2, key = lambda p: p.mass)
        
        # Assert the charge difference between the particles is 0 or 1. Different values are not supported.
        assert abs(p1.charge - p2.charge) in (0, 1), f"The charge difference must be of 0 or 1. |q1 - q2| = {abs(p1.charge - p2.charge)}"
        self.charged_decay = abs(p1.charge - p2.charge) == 1
        self.neutral_decay = not self.charged_decay
       
        # Identify which boson is included in the decay. Currently only W and H are supported. 
        self.boson = "W" if self.charged_decay else "H"
        
    
    def __eq__(self, other):
        same_particles = self.most_massive == other.most_massive and self.least_massive == other.least_massive
        
        return same_particles
        
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    
    def __repr__(self):
        return f"{self.most_massive.label} -> {self.least_massive} + {self.boson}"
    
    
    def __str__(self):
        return self.__repr__()
    
    
class Multiplet:
    
    def __init__(self, spin: int | Fraction, color: int, isospin: int | Fraction, hypercharge: int | Fraction, I: int):
        self.spin = spin
        self.color = color
        self.isospin = isospin
        self.hypercharge = hypercharge
        self.I = I
        
        self.dimension = int(2*isospin + 1)

        self._regular_names = {1: "Singlet", 2: "Doublet", 3: "Triplet", 4: "Quartet", 5: "Quintet"}
        
        if self.dimension in self._regular_names.keys():
            self._multiplet_name = self._regular_names[self.dimension]
            
        else: self._multiplet_name = f"{self.dimension}-plet"
        
        
    @property
    def flavor_eigenstates(self):
        """
        Returns a list of the flavor eigenstates that contitute the multiplet.
        """
        return [FlavorEigenstate(self, -self.isospin + i) for i in range(self.dimension)]
    
    
    def __len__(self):
        return self.dimension
    
    
    def __eq__(self, other):
        same_spin = self.spin == other.spin
        same_color = self.color == other.color
        same_isospin = self.isospin == other.isospin
        same_hypercharge = self.hypercharge == other.hypercharge
        same_I = self.I == other.I
        return same_spin and same_color and same_hypercharge and same_isospin and same_I
    
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    
    def __hash__(self):
        return hash((self.spin, self.color, self.isospin, self.hypercharge, self.I))
    
    
    def __iter__(self):
        return iter(self.flavor_eigenstates)
    
    
    def __getitem__(self, index):
        return self.flavor_eigenstates[index]
    
    
    def __repr__(self):
        return f"SU(2) {self._multiplet_name} ({self.spin}, {self.color}; {self.isospin}, {self.hypercharge}, {self.I})"
    
    
    def __str__(self):
        return self.__repr__()
    
    
class FlavorEigenstate:
    
    def __init__(self, parent_multiplet: Multiplet, isospin_projection: int | Fraction):
        
        self.parent_multiplet = parent_multiplet
        
        self.spin = parent_multiplet.spin
        self.color = parent_multiplet.color
        self.isospin = parent_multiplet.isospin
        self.hypercharge = parent_multiplet.hypercharge
        self.I = parent_multiplet.I
        self.isospin_projection = isospin_projection
        
        
        self.charge = self.hypercharge + self.isospin_projection
        
        
    def __eq__(self, other):
        same_multiplet = self.parent_multiplet == other.parent_multiplet
        same_projection = self.isospin_projection == other.isospin_projection
        
        return same_multiplet and same_projection
    
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    
    def __hash__(self):
        return hash((
            self.spin, 
            self.color, 
            self.isospin, 
            self.hypercharge, 
            self.I, 
            self.isospin_projection
            ))
    
        
    def __repr__(self):
        # return f"({self.spin}, {self.color}; {self.isospin}, {self.hypercharge}, {self.I}, {self.isospin_projection}) from SU(2) {self.parent_multiplet._multiplet_name}"
        return f"({self.isospin}, {self.hypercharge}, {self.I}, {self.isospin_projection})"
        
        
    def __str__(self):
        return self.__repr__()
        

class ParticleAssignment(UserDict):
    """
    Bidirectional One to Many Dictionary relating MassEigenstates to FlavorEigenstates.
    Just a normal dictionary with a inverse() method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def __setitem__(self, key, value):
        
        if isinstance(value, set):
            value = value
            
        elif isinstance(value, (list, tuple)):
            value = set(value)
            
        else: value = {value}
        
        super().__setitem__(key, value)
        
    
    def __delitem__(self, key):
        super().__delitem__(key)
    
        
    def __repr__(self):
        return super().__repr__()
    
   
    @property 
    def inverse(self):
        
        inv_map = {}
        
        for mass_eigentate, flavor_eigenstates in self.data.items():
            
            for flavor_eigenstate in flavor_eigenstates:
                
                if flavor_eigenstate not in inv_map: 
                    inv_map[flavor_eigenstate] = set()
                
                inv_map[flavor_eigenstate].add(mass_eigentate)

        return inv_map
    
class Model:
    
    def __init__(self, multiplets: List[Multiplet]):
        self.multiplets = multiplets
        
        
    def generate_initial_mappings(self, particles: List[Particle]) -> List[ParticleAssignment]:
        """
        Returns a list of all possible ways of mapping a mass eigenstate to a single flavor eigenstate
        """
        
        flavor_eigenstates = sorted(
            [f for multiplet in self.multiplets for f in multiplet], key=lambda f: f.isospin_projection)
        
        # 1. Separate in (spin, color, charge) to avoid doing more permutations than necessary
        spincolorcharges = set((p.spin, p.color, p.charge) for p in particles)
        
        possible_1to1s = []
        
        for spincolorcharge in spincolorcharges:
            
            grouped_particles = [p for p in particles if (p.spin, p.color, p.charge) == spincolorcharge]
            grouped_flavor_eigenstates = [fe for fe in flavor_eigenstates if (fe.spin, fe.color, fe.charge) == spincolorcharge]
            permutations = itertools.permutations(grouped_flavor_eigenstates)
            
            # All permutations of ways to relate the mass eigenstates to the flavor eigenstates
            possible_assignment = [ParticleAssignment(
                {p: fe for p, fe in zip(grouped_particles, permutation)}) for permutation in permutations]
            
            possible_1to1s.append(possible_assignment)
        
        # 2. Cartesian product of the assignments of different unbroken QNs
        all_possible_1to1s = [
            ParticleAssignment({k: v for d in combo for k, v in d.items()}) for combo in itertools.product(*possible_1to1s)]
            
        return all_possible_1to1s
    
    
    def build_connections(self, assignment: ParticleAssignment, all_decays: List[DecayChannel]) -> ParticleAssignment:
        """
        Uses the neutral decays to determine all the connections between mass and flavor eigenstates.
        """
        
        final_assignment = ParticleAssignment({p: set(f) for p, f in assignment.items()})
        neutral_decays = [decay for decay in all_decays if decay.neutral_decay]
        
        for decay in neutral_decays:
            final_assignment[decay.p1].update(assignment[decay.p2])
            final_assignment[decay.p2].update(assignment[decay.p1])
            
        return final_assignment
    
    
    def is_consistent(self, assignment: ParticleAssignment, all_decays: List[DecayChannel], where_failed: bool = False):
        """
        For a given assignment to be deemed consistent, the following checks need to be True:
        
        1. Mandatory Charged Decay Check: For every multiplet, if two flavor eigenstates differ in charge
        by one, a charged decay must exist between them. Therefore, the corresponding mass 
        eigenstates must also be linked by this decay.
        
        2. Forbidden Charged Decay Check: If there is a charged decay between two mass eigenstates,
        check if each is assigned to AT LEAST ONE flavor eigenstate from the same multiplet. 
        If the answer negative, then the configuration is invalid.
        
        3. Isospin Check: There can only be a valid neutral decay between two mass eigenstates
        if each mass eigenstate has AT LEAST ONE flavor eigenstate that differs from
        AT LEAST ONE flavor eigenstate connected to the other mass eigenstate by 1/2 in isospin 
        """
        
        # 1. Mandatory Charged Decay Check
        # Note for future: This might be too stringent. If even ONE decay does not appear in the
        # list all_decays, then this is considered inconsistent. Could it be the case that one
        # does not see the decay channel because it is very small? Then, how to loosen up this 
        # condition a bit? Maybe consider if any of the decay happens, or "most".
        for multiplet in self.multiplets:
            for f1, f2 in itertools.pairwise(multiplet):
                for p1, p2 in itertools.product(assignment.inverse[f1], assignment.inverse[f2]):
                    if DecayChannel(p1, p2) not in all_decays: 
                        return False
                    
        # 2. Forbidden Charged Decay
        charged_decays = [decay for decay in all_decays if decay.charged_decay]
        
        charged_violations = []
        
        for decay in charged_decays:
            
            charged_violation = True
            
            p1, p2 = (decay.p1, decay.p2)
            
            for f1, f2 in itertools.product(assignment[p1], assignment[p2]):
                # If at least one flavor eigenstate from one particle belongs to at least one 
                # flavor eigenstate from the other, then there is no charged violation.
                if f1.parent_multiplet == f2.parent_multiplet: charged_violation = False
            
            charged_violations.append(charged_violation)    
        
        if any(charged_violations): 
            return False        
                    
        # 3. Isospin Check
        # Note: Need to be strict with the type allowed in isospins for FlavorEigenstates,
        # Always Fractions, otherwise this will not work.
        neutral_decays = [decay for decay in all_decays if decay.neutral_decay]
        
        neutral_violations = []
        
        for decay in neutral_decays:
            
            neutral_violation = True
            
            p1, p2 = (decay.p1, decay.p2)
            
            for f1, f2 in itertools.product(assignment[p1], assignment[p2]):
                
                if abs(f1.isospin - f2.isospin) == Fraction(1,2): neutral_violation = False
                
            neutral_violations.append(neutral_violation)
        
        if any(neutral_violations): 
            return False
        
        return True
            
        
    def all_valid_assignments(self, particles: List[Particle], all_decays: List[DecayChannel], where_failed: bool = False) -> List[ParticleAssignment]:
       
        assignments: List[ParticleAssignment] = []
        
        # 1. Find all possible 1 to 1 connections between MassEigenstates and FlavorEigenstates. 
        initial_assignments: List[ParticleAssignment] = self.generate_initial_mappings(particles)
        
        # 2. Find the rest of the connections using neutral decays
        connected_assignments: List[ParticleAssignment] = []
        for assignment in initial_assignments:
            connected_assignments.append(self.build_connections(assignment, all_decays))
        
        # 3. Use consistency checks to eliminate certain assignments
        for assignment in connected_assignments:
            if self.is_consistent(assignment, all_decays, where_failed):
                assignments.append(assignment)
        
        # 4. Remove duplicates.
        # Could perhaps go a step further and classify assignments as equal if they end up generating the same model
        # e.g., if only the index I changes between different multiplets, a renaming of particles is completely equivalent.
        unique_assignments: List[ParticleAssignment] = []
        for a in assignments:
            if a not in unique_assignments:
                unique_assignments.append(a)
        
        return unique_assignments
    
    
    def __getitem__(self, index):
        return self.multiplets[index]
    
    
    def __eq__(self, other):
        return set(self.multiplets) == set(other.multiplets)
    
    
    def __hash__(self):
        return hash(frozenset(self.multiplets))
    
    
    def __repr__(self):
        return f"{tuple(sorted([m.dimension for m in self.multiplets], reverse=True))}"
    

class ModelsBuilder:
    
    def __init__(self, particles: List[Particle], all_decays: List[DecayChannel]):
        self.particles = particles
        self.all_decays = all_decays
        
        # Unbroken quantum numbers are spin, color, and charge: (s, c, q).
        self.unbroken_qns = [(p.spin, p.color, p.charge) for p in particles]
        
        self.grouped_spincolor: defaultdict = defaultdict(list)
        for spin, color, charge in self.unbroken_qns:
            self.grouped_spincolor[(spin, color)].append(charge)
        
        
    def valid_charge_partitions(self, charges: List[int]) -> Set[Tuple[Tuple[int]]]:
        """
        Returns all valid charge partitions given a list of charges
        Example: charges = valid_charge_partitions([0, 0, 0, 1, 1])
        returns {
            ((0,), (0,), (0,), (1,), (1,)), 
            ((0,), (0,), (0, 1), (1,)), 
            ((0,), (0, 1), (0, 1))}
        """
        # In case they do not come sorted
        charges = sorted(charges)
        
        partitions = set()
        n = len(charges)
        
        def backtrack(index: int, current_partition: List[List[int]]):
            
            if index == n:
                frozen_partition: Tuple[Tuple[int]] = tuple(sorted(tuple(chain) for chain in current_partition))
                
                partitions.add(frozen_partition)
                
                return
            
            charge = charges[index]
            
            for chain in current_partition:
                if abs(chain[-1] - charge) == 1 and charge not in chain:
                    
                    chain.append(charge)
                    backtrack(index + 1, current_partition)
                    
                    chain.pop()
                    
            current_partition.append([charge])
            
            backtrack(index + 1, current_partition)
            
            current_partition.pop()
            
        backtrack(0, [])
        
        return partitions
    
    
    @property
    def all_valid_partitions(self) -> defaultdict:
        """
        Calculates valid_charge_partitions for each (spin, color) combination from the input.
        returns {(spin1, color1): charge_partitions1, (spin2, color2): charge_partitions2], ...}
        """
        all_partitions = defaultdict(list)
            
        for spincolor, charges in self.grouped_spincolor.items():
            all_partitions[spincolor] = self.valid_charge_partitions(charges)
            
        return all_partitions
    
    
    def assign_model(self, spin: int | Fraction, color: int, charge_partition: Tuple[Tuple[int]]) -> Model:
        """
        A charge partition has a 1 to 1 correspondence to a Model.
        Given something like ((0,), (0, 1), (0, 1)) one can spot one SU(2) singlet and two SU(2) doublets,
        as well as their hypercharges.
        """
        
        multiplets: List[Multiplet] = []
        
        for sequence in charge_partition:
            
            I: int = 0
            
            dimension: int = len(sequence)
            isospin: Fraction = Fraction(dimension - 1, 2)
            
            smallest_charge: int | Fraction = min(sequence)
            # isospin projection m for the smallest charge is m = -j.
            # since we are using q = y + m => y = q - m, y = q_min + j.
            hypercharge: int | Fraction = smallest_charge + isospin
            
            # Update I until it is unique
            while Multiplet(spin, color, isospin, hypercharge, I) in multiplets:
                I += 1
                
            multiplets.append(Multiplet(spin, color, isospin, hypercharge, I))
        
        model = Model(multiplets)    
        
        if model.all_valid_assignments(self.particles, self.all_decays): return model
        
        
    @property    
    def all_valid_models(self) -> List[Model]:
        
        models: List[Model] = []
        for spincolor, partitions in self.all_valid_partitions.items():
            spin, color = spincolor
            models += [self.assign_model(spin, color, partition) for partition in partitions]
            
        # filter out invalid models.
        models = [model for model in models if model is not None]
        
        return models
    
    
    @property
    def all_valid_assignments(self) -> Dict[Model, List[ParticleAssignment]]:
        
        valid_assignments = {}
        
        for model in self.all_valid_models:
            valid_assignments[model] = model.all_valid_assignments(self.particles, self.all_decays)
            
        return valid_assignments
        
    
    
if __name__ == "__main__":
    
    particles = [
    Particle(label, Fraction(1,2), 1, charge, mass) for label, charge, mass in zip(
        "Chi00 Chi01 Chi02 Chi11 Chi12".split(),
        [0, 0, 0, 1, 1],
        [780.0, 1200.2, 1200.5, 1200.0, 1200.0]
    )]
    
    decays = [
    DecayChannel(*p) for p in [
        (particles[0], particles[1]),
        (particles[0], particles[2]),
        (particles[1], particles[2]),
        (particles[3], particles[0]),
        (particles[3], particles[1]),
        (particles[3], particles[2]),
        (particles[4], particles[0]),
        (particles[4], particles[1]),
        (particles[4], particles[2]),
    ]]
    
    model1 = Model([
    Multiplet(Fraction(1,2), 1, Fraction(1,2), Fraction(1,2), 1), 
    Multiplet(Fraction(1,2), 1, 0, 0, 1), 
    Multiplet(Fraction(1,2), 1, 0, 0, 2),
    Multiplet(Fraction(1,2), 1, 0, 1, 1)
    ])
    
    model2 = Model([
    Multiplet(Fraction(1,2), 1, Fraction(1,2), Fraction(1,2), 1), 
    Multiplet(Fraction(1,2), 1, Fraction(1,2), Fraction(1,2), 2), 
    Multiplet(Fraction(1,2), 1, 0, 0, 1),
    ])

    # print(model1)
    # print(model2.all_valid_assignments(particles, decays))
    
    solver = ModelsBuilder(particles, decays)
    
    print(solver.all_valid_assignments)
