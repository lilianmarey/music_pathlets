if __name__ == "__main__":
    from processing import (
        build_XY,
        build_tag_rankings,
        build_colistening,
        build_events_candidates,
        build_trajectories,
    )
    from modeling import (
        compute_subpaths_counter,
        compute_pathlet_dictionary,
        compute_pathlet_embedding,
    )
    from evaluating import compute_scores
