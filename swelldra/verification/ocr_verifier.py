def verify_document_structure(doc):
    issues = []
    orders = sorted([b.reading_order for b in doc.blocks])
    if orders != list(range(len(orders))):
        issues.append("Reading order not contiguous")
    return {"valid": len(issues)==0, "issues": issues}
