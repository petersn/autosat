"""
autosat - Tools for making SAT instances
"""

import os, functools, itertools, time, json, hashlib, sqlite3, inspect
import numpy as np
from . import tseytin
import cvxopt.glpk

cvxopt.solvers.options["show_progress"] = False

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_tseytin.db")


class Var:
    def __init__(self, instance: "Instance", number: int):
        self.instance = instance
        self.number = number

    def get_polarity(self, polarity: bool):
        return self if polarity else ~self

    def __invert__(self):
        return Var(self.instance, -self.number)


class Instance:
    def __init__(self):
        self.variables = {}
        self.clauses = []
        self.constants = [None, None]

    def new_var(self, name=None):
        number = len(self.variables) + 1
        var = Var(self, number)
        self.variables[number] = var
        return var

    def new_vars(self, count: int, name=None):
        return [self.new_var(name) for _ in range(count)]

    def get_constant(self, truth: bool):
        assert isinstance(truth, bool)
        if self.constants[truth] is None:
            var = self.constants[truth] = self.new_var(name=("false", "true")[truth])
            self.add_clause(var.get_polarity(truth))
        return self.constants[truth]

    def add_clause(self, *variables):
        for variable in variables:
            assert variable.instance is self
        self.clauses.append([variable.number for variable in variables])


class ImpossibleInputsError(Exception):
    pass


class TseytinDB:
    def __init__(self):
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


db = TseytinDB()


def to_num(bits):
    return sum(x * 2**i for i, x in enumerate(bits))


class Function:
    def __init__(self, f, input_bit_count, output_bit_count):
        self.f = f
        self.input_bit_count = input_bit_count
        self.output_bit_count = output_bit_count
        self.clauses = self._compile()

    def _compile(self):
        total_bits = self.input_bit_count + self.output_bit_count
        behavior = bytearray(2**total_bits)
        for input_comb in itertools.product([0, 1], repeat=self.input_bit_count):
            try:
                r = self.f(*input_comb)
                if isinstance(r, int):
                    r = r,
                output_comb = tuple(r)
            except ImpossibleInputsError:
                continue # This will result in us preventing any input here.
            assert all(i in (0, 1) for i in output_comb)
            behavior[to_num(input_comb + output_comb)] = 1
        # TODO: When I switch to python3: behavior = bytes(behavior)
        behavior = bytes(behavior)
        table_hash = hashlib.sha256(behavior).hexdigest()
        cached = db.get_clauses(table_hash)
        if cached is not None:
            return cached
        print("Cache miss: %i -> %i (%s)" % (self.input_bit_count, self.output_bit_count, table_hash))
        print("Behavior counts:", len(behavior), behavior.count(b"\0"), behavior.count(b"\1"))
        start_time = time.time()
        t = tseytin.Tseytin()

        if total_bits > 15:
            print("Using length-iterative greedy solver.")
            t.heuristic_solve(total_bits, behavior.decode())
            all_clauses = [list(clause) for clause in t.heuristic_solution]
        elif total_bits > 12:
            print("Using greedy solver.")
            t.setup(total_bits, behavior.decode(), 7)
            t.compute_greedy_solution()
            all_clauses = [
                list(t.all_feasible[clause_index])
                for clause_index in t.greedy_solution
            ]
            print("Got clauses:", len(all_clauses))
        else:
            print("Using exact set-cover -> ILP solver.")
            literal_limit = 6
            mat_width = t.setup(total_bits, behavior.decode(), literal_limit)
            mat_height = behavior.count(b"\0")
            print("Memory consumption: %.2f GiB" % (mat_width * mat_height * 8 * 2**-30,))
            #mat_height = 2**total_bits - 2**self.input_bit_count
            constraint_matrix = np.zeros((mat_height, mat_width), dtype=np.float64)
            print("Python Mat: %ix%i" % constraint_matrix.shape)
            t.fill_matrix(tseytin.python_helper_size_t_to_double_ptr(constraint_matrix.ctypes.data))

            M = lambda x: cvxopt.matrix(np.array(x, dtype=np.float64))

            clause_costs = [
                1 + len(clause)
                for clause in t.all_feasible
            ]
            bounds = [-1] * mat_height
            status, result = cvxopt.glpk.ilp(
                c=M(clause_costs),
                G=cvxopt.matrix(constraint_matrix),
                h=M(bounds),
                B=set(range(mat_width)),
                options={"msg_lev": "GLP_MSG_OFF"},
            )
            # Find those clauses.
            all_clauses = [
                list(t.all_feasible[clause_index])
                for clause_index, x in enumerate(result)
                if x > 0.5
            ]

        db.set_clauses(table_hash, all_clauses)
        print("Compiled in: %.2fs" % (time.time() - start_time,))
        print("Solution:", all_clauses)
        return all_clauses

    def apply(self, instance, ins, outs):
        assert len(ins) == self.input_bit_count
        assert len(outs) == self.output_bit_count
        all_bits = {i + 1: b for i, b in enumerate(ins + outs)}
        for clause in self.clauses:
            instance.add_clause(*(all_bits[abs(v)].get_polarity(v > 0) for v in clause))

    def apply_conditional(self, builder, condition, ins, outs=None):
        if outs == None:
            outs = [builder.new_var() for _ in range(self.output_bit_count)]
        assert len(ins) == self.input_bit_count
        assert len(outs) == self.output_bit_count
        all_bits = {i + 1: b for i, b in enumerate(ins + outs)}
        for clause in self.clauses:
            builder.add_clause(
                [all_bits[ v] for v in clause if v > 0],
                [all_bits[-v] for v in clause if v < 0] + [condition],
            )
        for a, b in zip(ins, outs):
            builder.add_clause([condition, a], [b])
            builder.add_clause([condition, b], [a])


def sat(f):
    arg_spec = inspect.signature(f)
    input_count = len(arg_spec.parameters)
    assert input_count > 0, "Zero input functions aren't supported right now."

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
        except ImpossibleInputsError:
            continue
    sat_func = Function(f, input_count, output_count)

    @functools.wraps(f)
    def wrapper(*args, outs=None):
        instances = {arg.instance for arg in args if not isinstance(arg, bool)}
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

