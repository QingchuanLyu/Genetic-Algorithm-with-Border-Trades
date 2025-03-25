import random


# Function to calculate the overlap between two fragments
"""
def find_overlap(a, b):
    max_overlap = 0
    for i in range(1, len(a) + 1):
        if b.startswith(a[-i:]):
            max_overlap = i
    return max_overlap
"""
def find_overlap(a, b):
    max_overlap = 0
    for i in range(len(a), 1 + 1):
        if b.startswith(a[-i:]):
            max_overlap = i
            break
    return max_overlap

# Function to calculate the supersequence length
"""
chromosome: A list of indices representing the order in which fragments should be arranged.

Example: [0, 2, 1] means fragments should be combined in the order: fragments[0], fragments[2], fragments[1].
fragments: A list of strings (e.g., DNA sequences, substrings, etc.).

Example: ["AGT", "GTCA", "TCAG"].
"""
def supersequence_length(chromosome, fragments):
    merged_seq = fragments[chromosome[0]]
    for i in range(1, len(chromosome)):
        overlap = find_overlap(merged_seq, fragments[chromosome[i]])
        merged_seq += fragments[chromosome[i]][overlap:]
    return len(merged_seq)


# Initialize population
def initialize_population(size, num_fragments):
    population = [random.sample(range(num_fragments), num_fragments) for _ in range(size)]
    return population


# Perform crossover
"""
This ensures that the offspring is a valid permutation (no duplicates).
Example: If parent2 = [D, A, E, B, C], the second part is [D, E, C]
"""
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
    return child


# Perform mutation
def mutate(chromosome):
    i, j = random.sample(range(len(chromosome)), 2)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


# Genetic Algorithm
def genetic_algorithm(fragments, population_size=100, generations=100):
    num_fragments = len(fragments)
    population = initialize_population(population_size, num_fragments)

    for generation in range(generations):
        # Evaluate fitness for each chromosome
        fitness_scores = [
            (chromosome, supersequence_length(chromosome, fragments))
            for chromosome in population
        ]
        fitness_scores.sort(key=lambda x: x[1])  # Sort by sequence length

        # Select the top-performing chromosomes
        next_generation = [chromosome for chromosome, _ in fitness_scores[:population_size // 2]]

        # Generate offspring through crossover
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation probability
                mutate(child)
            next_generation.append(child)

        population = next_generation

    # Return the best solution
    best_chromosome = min(population, key=lambda chrom: supersequence_length(chrom, fragments))
    return best_chromosome, supersequence_length(best_chromosome, fragments)


# DNA fragments
fragments = [
    "ATTAGACCTG",
    "CCTGCCGGAA",
    "AGACCTGCCG",
    "GCCGGAATAC"
]

# Run the genetic algorithm
best_solution, length = genetic_algorithm(fragments)
print("Best Solution Order:", best_solution)
print("Shortest Supersequence Length:", length)
