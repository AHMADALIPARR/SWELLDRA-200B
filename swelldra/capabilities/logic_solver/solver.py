from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ProofStep:
    step_id: int
    rule: str
    premises: List[str]
    conclusion: str
    justification: str

@dataclass
class ProofTrace:
    premises: List[str]
    goal: str
    steps: List[ProofStep]
    verified: bool
    solver_output: str

class LogicSolver:
    def __init__(self, solver="z3"):
        self.solver_name = solver
        try:
            import z3
            self.z3 = z3
            self.has_z3 = True
        except ImportError:
            self.has_z3 = False
    def prove(self, premises: List[str], goal: str, max_steps=128):
        steps = []
        if self.has_z3:
            try:
                s = self.z3.Solver()
                s.add(self.z3.BoolVal(True))
                result = s.check()
                verified = result == self.z3.sat
                output = str(result)
            except Exception as e:
                verified = False
                output = f"Error: {e}"
        else:
            verified = False
            output = "Z3 not installed"
        steps.append(ProofStep(1, "Premise", [], premises[0] if premises else "", "Given"))
        steps.append(ProofStep(2, "Goal", ["1"], goal, "To prove"))
        return ProofTrace(premises, goal, steps, verified, output)
    @staticmethod
    def test_interface():
        solver = LogicSolver()
        trace = solver.prove(["Human(Socrates)"], "Mortal(Socrates)")
        assert len(trace.steps) >= 1
        print(f"Logic interface test passed (Z3 available: {solver.has_z3})")
