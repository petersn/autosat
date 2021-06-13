// Tseytin transformations

#include <cstdio>
#include <cassert>
#include <iostream>
#include "autosat_tseytin.h"

bool do_logging = false;

static inline int popcount(uint64_t x) {
    // Both clang and gcc recognize the following, and will emit a popcnt instruction
    // at -msse4 and above, the same as we'd get with __builtin_popcountl.
    int result = 0;
    while (x) {
        x &= x - 1;
        result++;
    }
    return result;
}

static uint64_t int_pow(uint64_t base, int exponent) {
    uint64_t result = 1;
    for (int i = 0; i < exponent; i++)
        result *= base;
    return result;
}

static bool test_clause_behavior(const Clause& c, uint64_t setting) {
    for (int lit : c) {
        bool negate = lit < 0;
        int var = lit < 0 ? (-lit) - 1 : lit - 1;
        bool bit = (setting >> var) & 1;
        if (bit != negate)
            return true;
    }
    return false;
}

static bool test_compatible(uint64_t setting_max, const Clause& c, char* behavior) {
    for (uint64_t bit_setting = 0; bit_setting < setting_max; bit_setting++)
        if (behavior[bit_setting] and not test_clause_behavior(c, bit_setting))
            return false;
    return true;
}

static bool test_useful(
    uint64_t setting_max,
    const Clause& c,
    const std::vector<uint64_t>& settings_to_rule_out
) {
    for (uint64_t bit_setting : settings_to_rule_out)
        if (not test_clause_behavior(c, bit_setting))
            return true;
    return false;
}

static bool test_compatible_and_useful(
    uint64_t setting_max,
    const Clause& c,
    char* behavior,
    const std::vector<uint64_t>& settings_to_rule_out
) {
    return test_useful(setting_max, c, settings_to_rule_out) && test_compatible(setting_max, c, behavior);
}

int Tseytin::setup(int bits, char* behavior, int literal_limit) {
    // Find every matching clause.
    uint64_t clause_count_max = int_pow(3, bits);
    uint64_t setting_max = int_pow(2, bits);

    for (uint64_t clause_desc = 0; clause_desc < clause_count_max; clause_desc++) {
        // Compute the size.
        int lit_count = 0;
        uint64_t t = clause_desc;
        for (int i = 0; i < bits; i++) {
            int trit = t % 3;
            t /= 3;
            if (trit != 0)
                lit_count++;
        }
        if (lit_count > literal_limit)
            continue;

        Clause c;
        t = clause_desc;
        for (int i = 0; i < bits; i++) {
            int trit = t % 3;
            t /= 3;
            if (trit == 1)
                c.push_back(i + 1);
            if (trit == 2)
                c.push_back(-(i + 1));
        }
        if (test_compatible(setting_max, c, behavior))
            all_feasible.emplace_back(std::move(c));
    }

    for (uint64_t i = 0; i < setting_max; i++)
        if (not behavior[i])
            settings_to_rule_out.push_back(i);

    // TODO: Swap these variable names. They're backwards!
    int mat_height = all_feasible.size();
    int mat_width = settings_to_rule_out.size();

    if (do_logging) {
        std::cout << "Memory for approximate solvers: " << (((double)mat_width * (double)mat_height / 8) / 1024.0 / 1024.0 / 1024.0) << " GiB" << std::endl;
        std::cout << "C++ Mat:    " << mat_width << "x" << mat_height << std::endl;
    }

    return all_feasible.size();
}

void Tseytin::fill_matrix(double* buffer) {
    int mat_height = all_feasible.size();
    int mat_width = settings_to_rule_out.size();

    // Fill in the coverage matrix.
    for (int i = 0; i < mat_height; i++)
        for (int j = 0; j < mat_width; j++)
            buffer[i + j * mat_height] = test_clause_behavior(all_feasible[i], settings_to_rule_out[j]) - 1;

    if (do_logging)
        std::cout << "Matrix filled." << std::endl;
}

static inline void set_bit(uint64_t* data, int bit_index) {
    data[bit_index / 64] |= 1ull << (bit_index % 64);
}

static inline void clear_bit(uint64_t* data, int bit_index) {
    data[bit_index / 64] &= ~(1ull << (bit_index % 64));
}

static inline bool test_bit(uint64_t* data, int bit_index) {
    return data[bit_index / 64] & (1ull << (bit_index % 64));
}

int Tseytin::heuristic_solve(int bits, char* behavior) {
    if (do_logging)
        std::cout << "Heuristic solve. Bits: " << bits << std::endl;

    // Find every matching clause.
    uint64_t setting_max = int_pow(2, bits);

    std::vector<std::vector<Clause>> positive_clauses_by_length(bits + 1);

    for (uint64_t clause_desc = 0; clause_desc < setting_max; clause_desc++) {
        Clause c;
        for (int i = 0; i < bits; i++)
            if ((clause_desc >> i) & 1)
                c.push_back(i + 1);
        positive_clauses_by_length[c.size()].emplace_back(std::move(c));
    }

    for (uint64_t i = 0; i < setting_max; i++)
        if (not behavior[i])
            settings_to_rule_out.push_back(i);

    int mat_width = settings_to_rule_out.size();
    int row_width = (mat_width + 63) / 64;

    if (do_logging)
        std::cout << "  Mat width: " << mat_width << std::endl;

    // Find a greedy solution.
    std::vector<uint64_t> behavior_to_remove(row_width);
    for (uint64_t& x : behavior_to_remove)
        x = -1ull;
    for (int i = mat_width; i < row_width * 64; i++)
        clear_bit(&behavior_to_remove[0], i);

    int current_max_length = 0;
    while (true) {
        // Determine if we're done.
        int weight = 0;
        for (uint64_t x : behavior_to_remove)
            weight += popcount(x);
        if (weight == 0)
            break;
        current_max_length++;
        // Find all new feasible clauses of this length.

        // Filter down the settings_to_rule_out and behavior_to_remove mask.
        std::vector<uint64_t> new_settings_to_rule_out;
        for (uint64_t i = 0; i < settings_to_rule_out.size(); i++)
            if (test_bit(&behavior_to_remove[0], i))
                new_settings_to_rule_out.push_back(settings_to_rule_out[i]);
        if (do_logging) {
            std::cout << "    [" << heuristic_solution.size() << "] clause_length=" << current_max_length
                << " weight: " << weight << " length: " << settings_to_rule_out.size()
                << "->" << new_settings_to_rule_out.size() << std::endl;
        }

        settings_to_rule_out = std::move(new_settings_to_rule_out);
        mat_width = settings_to_rule_out.size();
        row_width = (mat_width + 63) / 64;
        // Find a greedy solution.
        behavior_to_remove.resize(row_width);
        for (uint64_t& x : behavior_to_remove)
            x = -1ull;
        for (int i = mat_width; i < row_width * 64; i++)
            clear_bit(&behavior_to_remove[0], i);

        std::vector<Clause> feasible_clauses_of_this_length;
        int polarity_max = int_pow(2, current_max_length);
        Clause c(current_max_length);
        if (do_logging) {
            std::cout << "      Filtering down from " <<positive_clauses_by_length[current_max_length].size()
            << " clauses by " << polarity_max << " polarities." << std::endl;
        }
        for (const Clause& d : positive_clauses_by_length[current_max_length]) {
            // Try all polarities of literals in this clause.
            for (int i = 0; i < polarity_max; i++) {
                for (int j = 0; j < current_max_length; j++)
                    c[j] = (i >> j) & 1 ? d[j] : -d[j];
                if (test_compatible_and_useful(setting_max, c, behavior, settings_to_rule_out))
                    feasible_clauses_of_this_length.push_back(c);
            }
        }

        if (feasible_clauses_of_this_length.size() == 0)
            continue;

        int mat_height = feasible_clauses_of_this_length.size();

        std::vector<uint64_t> bit_matrix(row_width * mat_height);

        // Fill in the coverage matrix.
        if (do_logging)
            std::cout << "      Fill in effort: " << mat_width << "x" << mat_height << std::endl;
        for (int i = 0; i < mat_height; i++) {
            uint64_t* row_ptr = &bit_matrix[row_width * i];
            for (int j = 0; j < mat_width; j++)
                if (not test_clause_behavior(feasible_clauses_of_this_length[i], settings_to_rule_out[j]))
                    set_bit(row_ptr, j);
        }
        if (do_logging)
            std::cout << "      Remaining bits:" << std::flush;

        while (true) {
            // Find the highest pop-count row.
            int best_row = -1, best_row_score = -1;
            for (int row = 0; row < mat_height; row++) {
                uint64_t* row_ptr = &bit_matrix[row_width * row];
                int row_score = 0;
                for (int i = 0; i < row_width; i++)
                    row_score += popcount(row_ptr[i] & behavior_to_remove[i]);
                if (row_score > best_row_score) {
                    best_row = row;
                    best_row_score = row_score;
                }
            }
            assert(best_row != -1);
            if (best_row_score == 0)
                break;

            uint64_t* row_ptr = &bit_matrix[row_width * best_row];
            heuristic_solution.push_back(feasible_clauses_of_this_length[best_row]);
            uint64_t any = 0;
            for (int i = 0; i < row_width; i++) {
                uint64_t new_value = behavior_to_remove[i] & ~row_ptr[i];
                behavior_to_remove[i] = new_value;
                any |= new_value;
            }
            if (not any)
                break;
            int count = 0;
            for (int i = 0; i < row_width; i++)
                count += popcount(behavior_to_remove[i]);
            if (do_logging)
                std::cout << " " << count << std::flush;
        }
        if (do_logging)
            std::cout << std::endl;
    }

    return 0;
}

double* python_helper_size_t_to_double_ptr(size_t x) {
    return reinterpret_cast<double*>(x);
}
