def compute_spl(success, path_len, shortest_path):
    if not success:
        return 0.0
    return (shortest_path / max(path_len, shortest_path))
