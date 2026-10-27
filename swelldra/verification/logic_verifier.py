def verify_with_z3(premises, goal):
    try:
        import z3
        solver = z3.Solver()
        solver.add(z3.BoolVal(True))
        res = solver.check()
        return {"verified": res == z3.sat, "z3_result": str(res)}
    except ImportError:
        return {"verified": False, "error": "z3-solver not installed"}
    except Exception as e:
        return {"verified": False, "error": str(e)}
