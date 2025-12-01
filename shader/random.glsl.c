// PCG Random Number Generator
struct PCGState {
    uint state;
    uint inc;
};

uint pcg_hash(inout PCGState rng) {
    uint oldstate = rng.state;
    rng.state = oldstate * 747796405u + rng.inc;
    uint word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

float pcg_float(inout PCGState rng) {
    return float(pcg_hash(rng)) / 4294967296.0;
}

// Initialize with unique seed per thread
void initPCG(inout PCGState rng, uint seed, uint sequence) {
    rng.state = 0u;
    rng.inc = (sequence << 1u) | 1u;
    pcg_hash(rng);
    rng.state += seed;
    pcg_hash(rng);
}
