autosat
=======

**Install with:** `pip install autosat`

A Python library for easily making SAT instances, by just decorating your functions.
The core tool is a decorator, `autosat.sat`, that automatically converts a function that takes bits as arguments into CNF.

```python
import autosat

inst = autosat.Instance()

@autosat.sat
def xor(x, y):
    return x ^ y

@autosat.sat
def full_adder(a, b, carry_in):
    r = a + b + carry_in
    return r & 1, (r & 2) >> 1

def add(a, b):
    assert len(a) == len(b)
    # True and False are okay to pass into @autosat.sat functions.
    carry = False
    result = []
    # Perform ripple-carry addition.
    for a_bit, b_bit in zip(a, b):
        sum_bit, carry = full_adder(a_bit, b_bit, carry)
        result.append(sum_bit)
    return result, carry

x, y = inst.new_vars(2)
z = xor(x, y)

# You can also add clauses manually.
inst.add_clause(x, y, ~z)

a = inst.new_vars(32)
b = inst.new_vars(32)
c, overflow_bit = add(a, b)

# Gives the list of clauses like: [[-1, -2, -3], [1, 2, -3], [1, -2, 3], ...]
print(inst.clauses)

# Gives a string like: "p cnf 132 454\n-1 -2 -3 0\n1 2 -3 0\n1 -2 3 0\n ..."
print(inst.to_dimacs())

# If you have pysat installed (pip install python-sat) you can also ask it for solutions:
model = inst.solve(
    solver_name="glucose4",
    decode_model=False,
)
# Gives output like: {1: False, 2: False, 3: False, ...} or the string "UNSAT"
print(model)

# Gives the value of the variable in the model.
print("x =", x.decode(model))

# Decodes the numerical value that a list of variables takes on in the model.
# Will print 0 in this case, because a = 0x00000000 in the solution.
print("a =", autosat.decode_number(a, model))
```

The logic inside of a decorated function can be any arbitrary Python, as the arguments are simply either 0 or 1.
Specifically, the decorated function is called at every possible combination of inputs, and a lookup-table is built.

The next step is to convert this lookup-table into CNF clauses to implement the given functionality.
To do this, the lookup-table is converted into a set cover instance (where each possible input/output pair to rule out is an element of the set, and each clause is a subset that rules out some input/output pairs), which is solved either heuristically or via integer linear programming if it's small enough.
The solution is written to a persistent cache automatically, so the decorator should be fast on future invocations.
This technique is tractible if the number of input + output bits of a decorated function is at most about 18ish.

You can also declare that some inputs to a function should be completely ruled out:

```python
@autosat.sat
def mux_three(address_high, address_low, x, y, z):
    address = 2 * address_high + address_low
    if address == 3:
        raise autosat.ImpossibleInputsError()
    return [x, y, z][address]
```

Alternatively, you can say that you don't care about the output values for some input values by raising `autosat.DontCare()`.

You can request that a function reuse bits, if you'd like to constrain those bits in multiple ways:

```python
@autosat.sat
def crummy_hash(a, b, c, d):
    for _ in range(8):
        a ^= b | (c & d)
        a, b, c, d = b, c, d, a
    return a, b, c, d

input0 = inst.new_vars(4)
input1 = inst.new_vars(4)

# We'd like to constrain that input0 and input1 both hash to the same value.
hash_result = crummy_hash(*input0)
# Reuse the same result bits, thus forcing equality.
crummy_hash(*input1, outs=hash_result)
```

License
-------

All code here is licensed under CC0 (public domain).

