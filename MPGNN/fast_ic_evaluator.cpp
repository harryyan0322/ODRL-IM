// Fast weighted-cascade IC evaluator for large formal baseline graphs.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Edge {
    int target;
    float probability;
};

static std::vector<int> parse_seeds(const std::string& value) {
    std::vector<int> seeds;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) seeds.push_back(std::stoi(token));
    }
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    return seeds;
}

static std::vector<int> read_seeds_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open seed file: " + path);
    std::vector<int> seeds;
    int seed;
    while (input >> seed) seeds.push_back(seed);
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    return seeds;
}

int main(int argc, char** argv) {
    std::string graph_path;
    std::string seed_text;
    std::string seed_file;
    int trials = 1000;
    uint64_t random_seed = 2026;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--graph" && index + 1 < argc) graph_path = argv[++index];
        else if (argument == "--seeds" && index + 1 < argc) seed_text = argv[++index];
        else if (argument == "--seeds-file" && index + 1 < argc) seed_file = argv[++index];
        else if (argument == "--trials" && index + 1 < argc) trials = std::stoi(argv[++index]);
        else if (argument == "--seed" && index + 1 < argc) random_seed = std::stoull(argv[++index]);
    }
    if (graph_path.empty() || (seed_text.empty() && seed_file.empty()) || trials <= 0) {
        std::cerr << "Usage: fast_ic_evaluator --graph PATH "
                  << "(--seeds A,B | --seeds-file PATH) --trials N --seed S\n";
        return 2;
    }

    std::ifstream input(graph_path);
    if (!input) throw std::runtime_error("Unable to open graph: " + graph_path);
    std::vector<std::pair<int, int>> raw_edges;
    std::set<std::pair<int, int>> undirected_edges;
    std::vector<int> indegree;
    int maximum = -1;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::stringstream fields(line);
        int source, target;
        if (!(fields >> source >> target)) continue;
        if (source == target) {
            maximum = std::max(maximum, source);
            continue;
        }
        maximum = std::max(maximum, std::max(source, target));
        if (static_cast<int>(indegree.size()) <= maximum) indegree.resize(maximum + 1, 0);
        undirected_edges.emplace(std::min(source, target), std::max(source, target));
    }
    for (const auto& edge : undirected_edges) {
        raw_edges.emplace_back(edge.first, edge.second);
        raw_edges.emplace_back(edge.second, edge.first);
    }
    indegree.assign(maximum + 1, 0);
    for (const auto& edge : raw_edges) ++indegree[edge.second];
    const int node_count = maximum + 1;
    if (node_count <= 0) throw std::runtime_error("Graph contains no edges");
    std::vector<std::vector<Edge>> adjacency(node_count);
    for (const auto& edge : raw_edges) {
        adjacency[edge.first].push_back({edge.second, 1.0f / indegree[edge.second]});
    }
    raw_edges.clear();
    raw_edges.shrink_to_fit();

    const auto seeds = seed_file.empty() ? parse_seeds(seed_text) : read_seeds_file(seed_file);
    if (seeds.empty()) throw std::runtime_error("Seed set is empty");
    for (int seed : seeds) {
        if (seed < 0 || seed >= node_count) throw std::runtime_error("Seed is outside graph");
    }

    std::mt19937_64 generator(random_seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<uint32_t> active_epoch(node_count, 0);
    std::vector<int> queue(node_count);
    std::vector<int> spreads;
    spreads.reserve(trials);
    for (int trial = 0; trial < trials; ++trial) {
        const uint32_t epoch = static_cast<uint32_t>(trial + 1);
        int head = 0;
        int tail = 0;
        for (int seed : seeds) {
            active_epoch[seed] = epoch;
            queue[tail++] = seed;
        }
        while (head < tail) {
            const int source = queue[head++];
            for (const auto& edge : adjacency[source]) {
                if (active_epoch[edge.target] != epoch && uniform(generator) <= edge.probability) {
                    active_epoch[edge.target] = epoch;
                    queue[tail++] = edge.target;
                }
            }
        }
        spreads.push_back(tail);
    }

    std::cout << "{\"num_nodes\":" << node_count << ",\"num_edges\":";
    size_t edge_count = 0;
    for (const auto& edges : adjacency) edge_count += edges.size();
    std::cout << edge_count << ",\"spreads\":[";
    for (size_t index = 0; index < spreads.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << spreads[index];
    }
    std::cout << "]}\n";
    return 0;
}
