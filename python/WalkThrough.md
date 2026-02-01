# Model Builder: Multiplets from Particles and Decays

## Input
- Lista completa de partículas BSM,
- Lista completa de decaimentos entre essas partículas e Higgs ou $W$.
  
## Output
- Todos os modelos que passam nos testes de consistência e
- Para cada modelo, todas as possibilidades de mapeamento entre autoestados de massa e autoestados de sabor.

## Classes definidas pelo código

- **Particle**: Autoestado de massa
  - _inicialização_: `label: str`, `spin: int | float`, `color: int`, `charge: int | float`, `mass: float`
  
- **DecayChannel**: Decaimentos. A partir das partículas de entrada determina se o decaimento é para o $W$ ou para o $h$.
  - _inicialização_: `p1: Particle`, `p2: Particle`
  
- **Multiplet**: Multipleto de $SU(2)$.
  - _inicialização_: `spin: int | float`, `color: int`, `isospin: int | float`, `hypercharge: int | float`, `index: int`
  - _métodos_: 
    - `flavor_eigenstates -> List[FlavorEigenstate]` -> retorna uma lista com os `FlavorEigenstate` que constituem o multipleto
     
        **Exemplo**: <pre>multiplet = Multiplet(1/2, 1, 1/2, 1/2, 0)
       multiplet.flavor_eigenstates
       \\\ [(1/2, 1/2, 0, -1/2), (1/2, 1/2, 0, 1/2)]
    </pre>

- **FlavorEigenstate**: Autoestado de sabor. 
  - _inicialização_: `parent_multiplet: Multiplet`, `isospin_projection: int | float`
  
- **Model**: Modelo, definido como uma lista de multipletos. Essa classe faz testes de consistência e determina todas as possibilidades de mapeamento entre autoestados de massa e de sabor que um dado modelo pode ter.
  - _inicialização_: `multiplets: List[Multiplet]`
  - _métodos_:
    - `generate_initial_mappings(particles: List[Particle]) -> List[ParticleAssignment]`: dado um conjunto de partículas, faz todos os mapeamentos 1 pra 1 entre partícula e autoestados de sabor do modelo.
   
        **Exemplo**: MultipletBuilder.ipynb

    - `build_connections(assignment: ParticleAssignment, all_decays: List[DecayChannel]) -> ParticleAssignment.`: cresce o mapeamento 1 pra 1 realizado pela função anterior usando os decaimentos neutros.

        **Exemplo**: MultipletBuilder.ipynb

    - `is_consistent(assignment: ParticleAssignment, all_decays: List[DecayChannel]) -> bool`: confere se o mapeamento completo, realizado pela função acima, é consistente, usando os seguintes critérios:
      1. **Critério de Decaimento carregado obrigatório**: Deve existir um decaimento carregado para cada vizinho de multipleto. Portanto, se dois autoestados de massa estão mapeados para cada um desses autoestados de sabor, deve existir um decaimento carregado entre eles. Se não existir, o mapeamento é inconsistente.
      2. **Critério de decaimento carregado proibido**: Caminho contrário do primeiro. Se existir um decaimento carregado entre dois autoestados de massa, cheque o mapeamento deles. Cada um deve ter PELO MENOS UM membro do mesmo multipleto. Caso contrário, mapeamento inconsistente.
      3. **Critério de isospin**: Se existir um decaimento neutro entre dois autoestados de massa, cheque o mapeamento deles. Cada um deve ter PELO MENOS UM membro de um multipleto que difira de isospin em 1/2 de PELO MENOS UM membro de outro multipleto.

        **Exemplo**: MultipletBuilder.ipynb

    - `all_valid_assignments(particles: List[Particle], all_decays: List[DecayChannel]) -> List[ParticleAssignment]`: usa todas as funções acima para construir uma lista de todos os mapeamentos válidos:
      1. Constrói todos os mapeamentos 1 pra 1 entre autoestados de massa e autoestados de sabor. 
      2. Expande os mapeamentos usando os decaimentos neutros
      3. Exclui os mapeamentos inconsistentes de acordo com os testes de consistência citados
      4. Remove mapeamentos duplicados.

        **Exemplo**: MultipletBuilder.ipynb

- **ParticleAssignment**: Dicionário bidirecional que mapeia autoestados de massa em autoestados de sabor e vice-versa.
  - _inicialização_: dicionário: `Dict[Particle, Set[FlavorEigenstate]]`
  - _métodos_: 
    - `inverse -> Dict[FlavorEigenstate, Set[Particle]]`: retorna o mapeamento inverso

        **Exemplo**: MultipletBuilder.ipynb

- **ModelsBuilder**: Solver para construir todas as possibilidades e aplicar os testes de consistência.
  - _inicialização_: `particles: List[Particle]`, `all_decays: List[DecayChannel]`
  - _métodos_:
    - `valid_charge_partitions(charges: List[int]) -> Set[Tuple[Tuple[int]]]`: dado um conjunto de cargas, usa backtracking para exaustar as possibilidades de criar partições de carga usando as cargas disponíveis.

        **Exemplo**: ``valid_charge_partitions([0, 0, 0, 1, 1]) \\{((0,), (0,), (0,), (1,), (1,)), ((0,), (0,), (0, 1), (1,)), ((0,), (0, 1), (0, 1))}``

        remove a necessidade de fazer qualquer outra conferência em relação a carga (tamanho máximo de multipleto, número máximo de n-pletos, etc.)

    - `all_valid_partitions -> Dict[(spin, color): charge_partitions]`: Calcula `valid_charge_partitions` das cargas de cada combinação de (spin, cor) das partículas do input.
  
       **Exemplo**: ``all_valid_partitions \\{(1/2, 1): {((0,), (0,), (0,), (1,), (1,)), ((0,), (0,), (0, 1), (1,)), ((0,), (0, 1), (0, 1))}}``
    
    - `assign_model(spin: int | float, color: int, charge_partition: Tuple[Tuple[int]]) -> Model`: Constrói um modelo a partir das partições de carga dado que cada partição corresponde a um modelo. Só retorna um modelo se ele for válido dado as partículas e decaimentos de entrada da classe. Essa validação é feita pelo método `all_valid_assignments(particles, decays)` da classe `Model`.

        **Exemplo**: No exemplo acima, a primeira partição ``((0,), (0,), (0,), (1,), (1,))`` corresponde a 5 singletos de $SU(2)$, cada um com hipercarga igual à carga. A última partição ``((0,), (0, 1), (0, 1))`` corresponde a 2 dubletos e 1 singleto, com hipercargas bem definidas também.
        ``assign_model(1/2, 1, ((0,), (0,), (0,), (1,), (1,))) \\ Model([5 singletos])``. MAS, esse modelo não passa por todas as condições, então o verdadeiro retorno da função é `[]`.

    - `all_valid_models -> List[Model]`: Para cada partição gerada por `all_valid_partitions`, gera um modelo usando `assign_model`.
    - `all_valid_assignments -> Dict[Model, List[ParticleAssignment]]`: Para cada modelo gera uma lista de todas as possibilidades de `ParticleAssignment`

    Passo a passo:

    1. Gera todas as partições de cargas possíveis dado as cargas disponíveis
    2. Para cada partição, gera um modelo
    3. Para cada modelo, gera todos os mapeamentos possíveis.
    