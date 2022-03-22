"""
autosat - Tools for making SAT instances
"""

import os
import time
import json
import hashlib
import functools
import itertools
import inspect
import warnings
import logging
import sqlite3
from typing import List, Dict, Tuple, Union, Iterable
from . import autosat_tseytin

PYSAT_IMPORT_WARNING = \
    "\x1b[91m----> To install pysat: pip install python-sat" \
    " (pysat is an unrelated module!) <----\x1b[0m"


class TseytinDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            create table if not exists tseytin
                (table_hash text, clauses text)
        """)

    def get_clauses(self, table_hash):
        for _, clauses in self.conn.execute("select * from tseytin where table_hash = ?", (table_hash,)):
            return json.loads(clauses)

    def set_clauses(self, table_hash, clauses):
        self.conn.execute(
            "insert into tseytin values(?, ?)",
            (table_hash, json.dumps(clauses)),
        )
        self.conn.commit()


GLOBAL_DB_PATH = None
global_db = None

def get_db():
    global global_db
    if global_db is not None:
        return global_db

    if GLOBAL_DB_PATH is None:
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_tseytin.db")
            global_db = TseytinDB(path)
            return global_db
        except sqlite3.OperationalError:
            warnings.warn(f"Couldn't make cache at: {path}")
            warnings.warn("We'll instead try at ./cache_tseytin.db -- if you don't want this set autosat.GLOBAL_DB_PATH")
            path = "./cache_tseytin.db"
    else:
        path = GLOBAL_DB_PATH

    global_db = TseytinDB(path)
    return global_db


class Var:
    def __init__(self, instance: "Instance", number: int, name):
        self.instance = instance
        self.number = number
        self.name = name

    def _get_polarity(self, polarity: bool):
        return self if polarity else ~self

    def _check_instance(self, other: Union["Var", bool]):
        assert isinstance(other, bool) or other.instance == self.instance, \
            "Variables must be from the same Instance"

    def __invert__(self):
        return Var(self.instance, -self.number, self.name)

    def __and__(self, other: Union["Var", bool]) -> "Var":
        self._check_instance(other)
        # Perform constant folding immediately.
        if isinstance(other, bool):
            return self if other else self.instance.get_constant(False)
        result = self.instance.new_var()
        self.instance.add_clause(self, ~result)
        self.instance.add_clause(other, ~result)
        self.instance.add_clause(~self, ~other, result)
        return result

    def __or__(self, other: Union["Var", bool]) -> "Var":
        self._check_instance(other)
        if isinstance(other, bool):
            return self if not other else self.instance.get_constant(True)
        result = self.instance.new_var()
        self.instance.add_clause(~self, result)
        self.instance.add_clause(~other, result)
        self.instance.add_clause(self, other, ~result)
        return result

    def __xor__(self, other: Union["Var", bool]) -> "Var":
        self._check_instance(other)
        if isinstance(other, bool):
            return self._get_polarity(not other)
        result = self.instance.new_var()
        self.instance.add_clause(self, other, ~result)
        self.instance.add_clause(self, ~other, result)
        self.instance.add_clause(~self, other, result)
        self.instance.add_clause(~self, ~other, ~result)
        return result

    def __rand__(self, other):
        return self.__and__(other)

    def __ror__(self, other):
        return self.__or__(other)

    def __rxor__(self, other):
        return self.__xor__(other)

    def make_equal(self, other: Union["Var", bool]):
        self._check_instance(other)
        if isinstance(other, bool):
            self.instance.add_clause(self if other else ~self)
        else:
            self.instance.add_clause(self, ~other)
            self.instance.add_clause(~self, other)

    def make_imply(self, other: Union["Var", bool]):
        self._check_instance(other)
        if isinstance(other, bool):
            if not other:
                self.make_equal(False)
        else:
            self.instance.add_clause(~self, other)

    def decode(self, model: Dict[int, bool]) -> bool:
        if self.number in model:
            return model[self.number]
        if -self.number in model:
            return not model[-self.number]
        return False
        #raise KeyError("Variable %r not found in model" % self)

    def __bool__(self):
        raise RuntimeError("You can't branch on a SAT variable in Python -- does what you're trying to do make sense?")

    def __repr__(self):
        if self.name is None:
            return "⟨%i⟩" % self.number
        return "⟨%i: %s⟩" % (self.number, self.name)


class Instance:
    def __init__(self):
        self.variables = {}
        self.clauses = []
        self.named_numbers = {}
        self.named_collections = {}
        self.constants = [None, None]

    def new_var(self, name=None) -> Var:
        number = len(self.variables) + 1
        var = Var(self, number, name)
        self.variables[number] = var
        if name is not None:
            if name not in self.named_collections:
                self.named_collections[name] = []
            self.named_collections[name].append(var)
        return var

    def new_vars(self, count: int, name=None, *, is_number=False) -> List[Var]:
        if is_number:
            assert name is not None, "The is_number flag changes how we decode the model, and is meaningless without a name"
            assert name not in self.named_numbers, f"Duplicate name for named number: {name}"
        variables = [self.new_var(name) for _ in range(count)]
        if is_number:
            self.named_numbers[name] = variables
        return variables

    def get_constant(self, truth: bool) -> Var:
        assert isinstance(truth, bool)
        if self.constants[truth] is None:
            var = self.constants[truth] = self.new_var(name=("false", "true")[truth])
            self.add_clause(var._get_polarity(truth))
        return self.constants[truth]

    def add_clause(self, *variables):
        for variable in variables:
            assert variable.instance is self
        self.clauses.append([variable.number for variable in variables])

    def to_dimacs(self) -> str:
        s = ["p cnf %i %i\n" % (len(self.variables), len(self.clauses))]
        for clause in self.clauses:
            s.append(" ".join(map(str, clause + [0])) + "\n")
        return "".join(s)

    def to_pysat_formula(self):
        try:
            import pysat.formula
        except ImportError:
            warnings.warn(PYSAT_IMPORT_WARNING)
            raise
        cnf = pysat.formula.CNF()
        cnf.from_clauses(self.clauses)
        return cnf

    def solve(self, solver_name="glucose4", decode_model=True):
        try:
            import pysat.solvers
        except ImportError:
            warnings.warn(PYSAT_IMPORT_WARNING)
            raise
        solver = pysat.solvers.Solver(name=solver_name, bootstrap_with=self.clauses)
        satisfiable = solver.solve()
        if not satisfiable:
            return "UNSAT"
        model = {abs(var): var > 0 for var in solver.get_model()}
        if decode_model:
            model = self.decode_model(model)
        return model

    def decode_model(self, model):
        results = {}
        for name, variables in self.named_collections.items():
            results[name] = [var.decode(model) for var in variables]
        for name, variables in self.named_numbers.items():
            results[name] = decode_number(variables, model)
        return results


def decode_number(variables: Iterable[Var], model: Dict[int, bool]) -> int:
    return sum(var.decode(model) * 2**i for i, var in enumerate(variables))

def _log2(x):
    assert x & (x - 1) == 0, "Number isn't a power of two"
    return x.bit_length() - 1

def behavior_table_to_clauses(
    behavior: bytes,
    use_cache=True,
    exact_solution_cost_function=None,
) -> List[List[int]]:
    total_bits = _log2(len(behavior))

    if use_cache:
        table_hash = hashlib.sha256(behavior).hexdigest()
        cached = get_db().get_clauses(table_hash)
        if cached is not None:
            return cached

    logging.info(f"Cache miss: bits={total_bits} hash={table_hash}")
    logging.info("Behavior counts: total=%i zero=%i one=%i" % (
        len(behavior), behavior.count(b"\0"), behavior.count(b"\1"),
    ))
    start_time = time.time()
    t = autosat_tseytin.Tseytin()

    if exact_solution_cost_function is None:
        # This required .decode() here is really annoying.
        # I should find a way to make SWIG let me pass bytes.
        t.heuristic_solve(total_bits, behavior.decode())
        all_clauses = [list(clause) for clause in t.heuristic_solution]
    else:
        logging.info("Using exact set-cover -> ILP solver.")

        import numpy as np
        import cvxopt.glpk
        cvxopt.solvers.options["show_progress"] = False

        literal_limit = 6
        mat_width = t.setup(total_bits, behavior.decode(), literal_limit)
        mat_height = behavior.count(b"\0")
        logging.info(f"Memory consumption: {mat_width * mat_height * 8 * 2**-30:.2f} GiB")
        constraint_matrix = np.zeros((mat_height, mat_width), dtype=np.float64)
        logging.info("Python Mat: %ix%i" % constraint_matrix.shape)
        t.fill_matrix(autosat_tseytin.python_helper_size_t_to_double_ptr(constraint_matrix.ctypes.data))

        M = lambda x: cvxopt.matrix(np.array(x, dtype=np.float64))

        clause_costs = list(map(exact_solution_cost_function, t.all_feasible))
        bounds = [-1] * mat_height
        status, result = cvxopt.glpk.ilp(
            c=M(clause_costs),
            G=cvxopt.matrix(constraint_matrix),
            h=M(bounds),
            B=set(range(mat_width)),
            options={"msg_lev": "GLP_MSG_OFF"},
        )

        all_clauses = [
            list(t.all_feasible[clause_index])
            for clause_index, x in enumerate(result)
            if x > 0.5
        ]

    if use_cache:
        db.set_clauses(table_hash, all_clauses)

    logging.info("Compiled in: %.2fs" % (time.time() - start_time))
    logging.info("Solution: %s" % all_clauses)
    return all_clauses

def sat_args_helper(
    input_count: int,
    output_count: int,
    bits_in: Iterable[Union[Var, bool]],
    bits_out: Union[None, Iterable[Union[Var, bool]]],
) -> Tuple[Instance, List[Var], List[Var]]:
    assert len(bits_in) == input_count, \
        f"Wrong number of input bits: expected {input_count}, got {len(bits_in)}"
    assert bits_out is None or len(bits_out) == output_count, \
        f"Wrong number of output bits: expected {output_count}, got {len(bits_out)}"

    instances = {bit.instance for bit in bits if not isinstance(bit, bool)}
    assert len(instances) != 0, "You can't have all arguments to an @autosat.sat function be bool," \
        " because then we don't know what instance to build into (try using instance.get_constant instead)"
    assert len(instances) == 1, "All variables into a function must come from the same instance"
    instance = instances.pop()

    if bits_out is None:
        bits_out = instance.new_vars(output_count)

    vars_in, vars_out = [
        [
            instance.get_constant(bit) if isinstance(bit, bool) else bit
            for bit in bits_list
        ]
        for bits_list in (bits_in, bits_out)
    ]
    return instance, vars_in, vars_out


class ImpossibleInputsError(Exception): pass

class DontCare(Exception): pass


class Function:
    def __init__(self, f, input_bit_count, output_bit_count, clauses):
        self.f = f
        self.input_bit_count = input_bit_count
        self.output_bit_count = output_bit_count
        self.clauses = clauses

    def call_underlying_function(self, *bits):
        try:
            return self.f(*bits)
        except DontCare:
            return (0,) * self.output_bit_count

    def __call__(self, *bits_in, outs=None):
        instance, vars_in, vars_out = sat_args_helper(
            self.input_bit_count, self.output_bit_count, bits_in, outs,
        )
        all_vars = {i + 1: v for i, v in enumerate(vars_in + vars_out)}
        for clause in self.clauses:
            instance.add_clause(*(
                all_vars[abs(literal)]._get_polarity(literal > 0)
                for literal in clause
            ))
        return vars_out

def sat(f=None, /, lazy=False, input_bits=None, output_bits=None, input_number=False, output_number=False):
    assert (not input_number) or input_bits is not None, \
        "If the function takes in its argument as a number, then you must specify the number of bits"
    assert (not output_number) or output_bits is not None, \
        "If the function returns its result as a number, then you must specify the number of bits"

    def decorator(f):
        return Function()

    # If f is None then we've been called like @autosat.sat(options=...), and need to *return* the decorator.
    # If f isn't None, then we've been called like @autosat.sat, and need to just immediately decorate.
    return decorator if f is None else decorator(f)

def sat(f):
    arg_spec = inspect.signature(f)
    input_count = len(arg_spec.parameters)
    assert input_count > 0, "Zero input functions aren't supported right now"

    output_count = 0
    promoting_single_bit = False
    for bit_pattern in itertools.product([0, 1], repeat=input_count):
        try:
            r = f(*bit_pattern)
            if isinstance(r, int):
                promoting_single_bit = True
                output_count = 1
                break
            output_count = len(r)
            break
        except (ImpossibleInputsError, DontCare):
            continue
    sat_func = Function(f, input_count, output_count)

    @functools.wraps(f)
    def wrapper(*args, outs=None):
        for arg in args:
            if not isinstance(arg, (Var, bool)):
                raise TypeError("Arguments to a @autosat.sat function must either be Var or bool")
        instances = {arg.instance for arg in args if not isinstance(arg, bool)}
        assert len(instances) != 0, "You can't have all arguments to an @autosat.sat function be bool," \
            " because then we don't know what instance to build into (try using instance.get_constant instead)"
        assert len(instances) == 1, "All variables into a function must come from the same instance"
        instance = instances.pop()
        args = [
            instance.get_constant(arg) if isinstance(arg, bool) else arg
            for arg in args
        ]
        if outs is None:
            outs = [instance.new_var() for _ in range(output_count)]
        sat_func.apply(instance, args, outs)
        if promoting_single_bit:
            bit, = outs
            return bit
        return outs

    return wrapper

@sat(lazy=True)
def full_adder(a, b, carry_in):
    r = a + b + carry_in
    return r & 1, (r & 2) >> 1

def add(a, b, return_final_carry=False):
    assert len(a) == len(b)
    carry = False
    result = []
    # Perform ripple-carry addition.
    for a_bit, b_bit in zip(a, b):
        sum_bit, carry = full_adder(a_bit, b_bit, carry)
        result.append(sum_bit)
    if return_final_carry:
        return result, carry
    return result

@sat(lazy=True)
def ternary_operator_gate(condition, a, b):
    return a if condition else b

@sat(lazy=True)
def sort_gate(a, b):
    return min(a, b), max(a, b)

def constrain_count(
    bits: List[Var],
    *,
    at_least=None,
    at_most=None,
    implementation="sorting",
    preserve_model_count=False,
):
    #assert implementation in {"sorting", "scan", "brute-force"}
    assert implementation == "sorting"
    assert (at_least is not None) or (at_most is not None), "Must constrain the count in some way"
    bits = list(bits)
    
    if implementation == "sorting":
        ...

    raise NotImplementedError
