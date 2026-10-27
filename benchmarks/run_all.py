from swelldra.capabilities.ocr.document_engine import OCRDocumentEngine
from swelldra.capabilities.sql_engine.engine import SQLEngine
from swelldra.capabilities.graph_engine.engine import KnowledgeGraph
from swelldra.capabilities.logic_solver.solver import LogicSolver
def run_benchmarks():
    print("Running SWELLDRA benchmarks...")
    OCRDocumentEngine.test_interface()
    SQLEngine.test_interface()
    KnowledgeGraph.test_interface()
    LogicSolver.test_interface()
if __name__ == "__main__":
    run_benchmarks()
