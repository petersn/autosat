// Tseytin transformations

#ifndef AUTOSAT_TSEYTIN_H
#define AUTOSAT_TSEYTIN_H

#include <cstdint>
#include <vector>

extern bool do_logging;

typedef std::vector<int> Clause;

struct Tseytin {
    std::vector<Clause> all_feasible;
    std::vector<uint64_t> settings_to_rule_out;
    std::vector<int> greedy_solution;
    std::vector<Clause> heuristic_solution;

    int setup(int bits, char* behavior, int literal_limit);
    void fill_matrix(double* buffer);

    int heuristic_solve(int bits, char* behavior);
};

double* python_helper_size_t_to_double_ptr(size_t x);

#endif
