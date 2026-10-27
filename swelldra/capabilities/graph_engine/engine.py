from dataclasses import dataclass
from typing import List, Dict
import networkx as nx

@dataclass
class Entity:
    id: str
    type: str
    properties: Dict

@dataclass
class Relation:
    source: str
    target: str
    type: str
    properties: Dict

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
    def add_entity(self, e: Entity):
        self.entities[e.id] = e
        self.graph.add_node(e.id, **e.properties, entity_type=e.type)
    def add_relation(self, r: Relation):
        self.graph.add_edge(r.source, r.target, type=r.type, **r.properties)
    def traverse(self, start_id: str, relation_types: List[str], max_hops=3):
        paths = []
        def dfs(current, path, hops):
            if hops > max_hops:
                return
            paths.append(list(path))
            for _, neighbor, data in self.graph.out_edges(current, data=True):
                if not relation_types or data.get('type') in relation_types:
                    if neighbor not in path:
                        dfs(neighbor, path + [neighbor], hops+1)
        dfs(start_id, [start_id], 0)
        return paths
    def check_invariants(self):
        issues = []
        for u, v, data in self.graph.edges(data=True):
            if u not in self.entities or v not in self.entities:
                issues.append(f"Orphan edge {u}->{v}")
        return {"valid": len(issues)==0, "issues": issues}
    @staticmethod
    def test_interface():
        kg = KnowledgeGraph()
        kg.add_entity(Entity("alice", "Person", {"name":"Alice"}))
        kg.add_entity(Entity("bob", "Person", {"name":"Bob"}))
        kg.add_relation(Relation("alice", "bob", "knows", {}))
        paths = kg.traverse("alice", ["knows"], max_hops=2)
        assert len(paths) >= 1
        print("Graph interface test passed")
