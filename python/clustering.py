import numpy as np
from typing import List, Union, Dict, Set


class Particle:
    
    def __init__(self, label: Union[int, str], mass: float, charge: int):
        self.label = label
        self.mass = mass
        self.charge = charge
        
    def __str__(self):
        return f"{self.label, self.mass, self.charge}"
    
    def __repr__(self):
        return self.__str__()


class DecayChannel:
    
    def __init__(self, particle1: Particle, particle2: Particle, branching_fraction: float):
        self.particle1 = particle1
        self.particle2 = particle2
        self.branching_fraction = branching_fraction
        
    @property
    def weight(self):
        mass1 = self.particle1.mass
        mass2 = self.particle2.mass
        return float(np.sqrt(self.branching_fraction * min(mass1 / mass2, mass2 / mass1)))
    
    def __str__(self):
        return f"({self.particle1.label} <-> {self.particle2.label}, BF = {self.branching_fraction})"
    
    def __repr__(self):
        return self.__str__()


def cluster_particles(decays: List[DecayChannel]):
    sorted_decay: List[DecayChannel] = sorted(decays, key = lambda x: x.weight, reverse=True)
    print(sorted_decay)
    print([decay.weight for decay in sorted_decay])
    clusters: Dict[Particle, Set[Particle]] = {}
    possible_singlet: Set[Particle] = set()
    
    while sorted_decay: # while the list is not empty
        decay = sorted_decay.pop(0)
        p1 = decay.particle1
        p2 = decay.particle2
        
        p1_in_clusters = p1 in clusters.keys()
        p2_in_clusters = p2 in clusters.keys()
        
        # Neither particle is clustered. Cluster them.
        if (not p1_in_clusters) and (not p2_in_clusters):
            clusters[p1] = {p1, p2}
            clusters[p2] = {p1, p2}
            
        # p1 is in a cluster, but p2 isnt.
        elif p1_in_clusters and (not p2_in_clusters):
            charge_conflict = False
            for particle in clusters[p1]:
                if p2.charge == particle.charge: 
                    charge_conflict = True
            
            # Can p2 join p1?
            if not charge_conflict:
                clusters[p1].add(p2)
                clusters[p2] = clusters[p1]
                
            # If there is a conflict, p2 may be a singlet
            else: possible_singlet.add(p2)
            
        # p2 is in a cluster, but p1 isnt.
        elif (not p1_in_clusters) and p2_in_clusters:
            charge_conflict = False
            for particle in clusters[p2]:
                if p1.charge == particle.charge: 
                    charge_conflict = True
                    
            # Can p1 join p2?
            if not charge_conflict:
                clusters[p2].add(p1)
                clusters[p1] = clusters[p2]
                
            # If there is a conflict, p1 may be a singlet
            else: possible_singlet.add(p1)
                
        # Both particles are already in clusters. Can the clusters be combined?
        elif p1_in_clusters and p2_in_clusters and clusters[p1] != clusters[p2]:
            charge_conflict = False
            for p3 in clusters[p1]:
                for p4 in clusters[p2]:
                    if p3.charge == p4.charge: charge_conflict = True
                
            union = clusters[p1].union(clusters[p2])    
                
            if not charge_conflict:
                for particle in clusters[p1]:
                    clusters[particle] = union
                for particle in clusters[p2]:
                    clusters[particle] = union
                    
        for particle in possible_singlet:
            if particle not in clusters.keys():
                clusters[particle] = {particle}
            
    return clusters
        
                

if __name__ == '__main__':
    
    from SDDM import SDDM
    
    sddm = SDDM(ms = 800, mu = 1200, y1 = 0.1, y2 = 0.2)
    masses = [float(mass) for mass in sddm.physical_masses]
    print(f"ms = {sddm.ms}, mu = {sddm.mu}, y1 = {sddm.y1}, y2 = {sddm.y2}")
    print(sddm.physical_masses)
    # print(sddm.tilde_BFW)
    
    particles = [Particle(label, mass, charge) for label, mass, charge in zip(range(5), masses, [0,0,0,1,1])]
    decay_channels = [
        DecayChannel(particles[0],particles[3], sddm.tilde_BFW[0,0]), 
        DecayChannel(particles[0],particles[4], sddm.tilde_BFW[1,0]), 
        DecayChannel(particles[1],particles[3], sddm.tilde_BFW[0,1]), 
        DecayChannel(particles[1],particles[4], sddm.tilde_BFW[1,1]), 
        DecayChannel(particles[2],particles[3], sddm.tilde_BFW[0,2]), 
        DecayChannel(particles[2],particles[4], sddm.tilde_BFW[1,2])
        ]
    
    # for channel in decay_channels:
    #     print(channel, channel.weight)
    print(cluster_particles(decay_channels))