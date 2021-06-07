// Tseytin transformations

#ifndef CRUX_TSEYTIN_H
#define CRUX_TSEYTIN_H

#include <cstdint>
#include <vector>

typedef std::vector<int> Clause;

struct Tseytin {
	std::vector<Clause> all_feasible;
	std::vector<uint64_t> settings_to_rule_out;
	std::vector<int> greedy_solution;
	std::vector<Clause> heuristic_solution;

	int setup(int bits, char* behavior, int literal_limit);
	void fill_matrix(double* buffer);
	void compute_greedy_solution();

	int heuristic_solve(int bits, char* behavior);
};

static double* python_helper_size_t_to_double_ptr(size_t x) {
	return reinterpret_cast<double*>(x);
}

#endif
